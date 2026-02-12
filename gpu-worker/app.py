from __future__ import annotations

import io
import os
import json
import uuid
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# 결과물 이름(프론트/백엔드가 이걸로 가져감)
GAUSSIANS_PLY_NAME = "gaussians.ply"

# nerfstudio 커맨드 (CUDA 필요)
# - ns-process-data images
# - ns-train splatfacto
# - ns-export gaussian-splat --export-gaussian-ply
#
# nerfstudio는 splatfacto/gaussian export 플로우가 있음.   [oai_citation:2‡Scuba Tech](https://www.scuba.tech/tools-and-resources/osiris-gsharc?utm_source=chatgpt.com)

app = FastAPI(title="CSRAI GPU Worker", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP: 일단 전체 허용 (심사 데모용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _job_paths(job_id: str):
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    work_dir = job_dir / "work"
    out_dir = job_dir / "output"
    meta_path = job_dir / "meta.json"
    return job_dir, input_dir, work_dir, out_dir, meta_path


def _read_meta(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    return json.loads(p.read_text("utf-8"))


def _write_meta(p: Path, meta: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")


def _run(cmd: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)


def _tail(s: str, n: int = 8000) -> str:
    return (s or "")[-n:]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/train")
async def train(images_zip: UploadFile = File(...)):
    """
    images_zip: zip 파일(내부에 jpg/png들이 들어있으면 됨)
    반환: job_id + status_url + gaussians_url
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir, input_dir, work_dir, out_dir, meta_path = _job_paths(job_id)

    input_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"job_id": job_id, "status": "queued"}
    _write_meta(meta_path, meta)

    # zip 저장/해제
    zip_bytes = await images_zip.read()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(input_dir)
    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = f"zip extract failed: {repr(e)}"
        _write_meta(meta_path, meta)
        raise HTTPException(400, detail="Invalid zip")

    # 이미지 폴더 찾기(중첩 zip 대비)
    # input_dir 아래에 이미지가 바로 있거나, 한 단계 폴더로 들어있을 수 있음
    def has_images(p: Path) -> bool:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return any((p / f).suffix.lower() in exts for f in os.listdir(p) if (p / f).is_file())

    images_dir = input_dir
    if not has_images(images_dir):
        # 1-depth 폴더 탐색
        for child in input_dir.iterdir():
            if child.is_dir() and has_images(child):
                images_dir = child
                break

    if not has_images(images_dir):
        meta["status"] = "failed"
        meta["error"] = "no images found in zip"
        _write_meta(meta_path, meta)
        raise HTTPException(400, detail="No images found")

    # 실제 학습 (동기 실행 MVP)
    meta["status"] = "running"
    _write_meta(meta_path, meta)

    # 1) ns-process-data images
    # work_dir/nerf 에 dataset 생성
    dataset_dir = work_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    cmd1 = [
        "ns-process-data", "images",
        "--data", str(images_dir),
        "--output-dir", str(dataset_dir),
    ]
    p1 = _run(cmd1)
    meta["cmd_process"] = " ".join(cmd1)
    meta["stdout_process"] = _tail(p1.stdout)
    meta["stderr_process"] = _tail(p1.stderr)
    if p1.returncode != 0:
        meta["status"] = "failed"
        meta["error"] = f"ns-process-data failed (code={p1.returncode})"
        _write_meta(meta_path, meta)
        return JSONResponse(meta, status_code=500)

    # 2) ns-train splatfacto
    # 모델 출력은 work_dir/outputs 아래 생김
    cmd2 = [
        "ns-train", "splatfacto",
        "--data", str(dataset_dir),
        "--output-dir", str(work_dir / "outputs"),
        "--max-num-iterations", os.environ.get("NS_ITERS", "7000"),
    ]
    p2 = _run(cmd2)
    meta["cmd_train"] = " ".join(cmd2)
    meta["stdout_train"] = _tail(p2.stdout)
    meta["stderr_train"] = _tail(p2.stderr)
    if p2.returncode != 0:
        meta["status"] = "failed"
        meta["error"] = f"ns-train failed (code={p2.returncode})"
        _write_meta(meta_path, meta)
        return JSONResponse(meta, status_code=500)

    # outputs 폴더에서 최신 config / checkpoint 찾기(간단히 가장 최근 run 사용)
    outputs_root = work_dir / "outputs"
    if not outputs_root.exists():
        meta["status"] = "failed"
        meta["error"] = "outputs dir not found"
        _write_meta(meta_path, meta)
        return JSONResponse(meta, status_code=500)

    # 가장 최근 수정된 run 디렉토리 찾기
    runs = [p for p in outputs_root.rglob("*") if p.is_dir()]
    runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    latest_run = runs[0] if runs else None
    if not latest_run:
        meta["status"] = "failed"
        meta["error"] = "no run directory found"
        _write_meta(meta_path, meta)
        return JSONResponse(meta, status_code=500)

    # 3) ns-export gaussian-splat (gaussians.ply)
    # export 폴더를 out_dir로 고정
    cmd3 = [
        "ns-export", "gaussian-splat",
        "--load-config", str(latest_run / "config.yml"),
        "--output-dir", str(out_dir),
        "--export-gaussian-ply",
    ]
    p3 = _run(cmd3)
    meta["cmd_export"] = " ".join(cmd3)
    meta["stdout_export"] = _tail(p3.stdout)
    meta["stderr_export"] = _tail(p3.stderr)
    if p3.returncode != 0:
        meta["status"] = "failed"
        meta["error"] = f"ns-export failed (code={p3.returncode})"
        _write_meta(meta_path, meta)
        return JSONResponse(meta, status_code=500)

    # out_dir 안에서 ply 찾기
    ply = None
    for cand in out_dir.rglob("*.ply"):
        # gaussian ply로 추정되는 파일 우선
        ply = cand
        break
    if not ply or not ply.exists():
        meta["status"] = "failed"
        meta["error"] = "gaussian ply not found after export"
        _write_meta(meta_path, meta)
        return JSONResponse(meta, status_code=500)

    # 통일 파일명으로 복사
    final_ply = out_dir / GAUSSIANS_PLY_NAME
    if ply.resolve() != final_ply.resolve():
        shutil.copyfile(ply, final_ply)

    meta["status"] = "done"
    meta["gaussians_ply"] = str(final_ply)
    _write_meta(meta_path, meta)

    base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    # 배포 시 PUBLIC_BASE_URL을 "https://xxxx.runpod.net" 같이 넣어라
    status_url = f"{base}/api/jobs/{job_id}" if base else f"/api/jobs/{job_id}"
    gaussians_url = f"{base}/api/jobs/{job_id}/gaussians.ply" if base else f"/api/jobs/{job_id}/gaussians.ply"

    return {
        "job_id": job_id,
        "status": "done",
        "status_url": status_url,
        "gaussians_ply_url": gaussians_url,
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job_dir, input_dir, work_dir, out_dir, meta_path = _job_paths(job_id)
    if not meta_path.exists():
        raise HTTPException(404, detail="job not found")

    meta = _read_meta(meta_path)
    ply = out_dir / GAUSSIANS_PLY_NAME
    meta_out = {
        "job_id": job_id,
        "status": meta.get("status", "unknown"),
        "error": meta.get("error"),
        "gaussians_ply_exists": ply.exists(),
    }
    return meta_out


@app.get("/api/jobs/{job_id}/gaussians.ply")
def download_gaussians(job_id: str):
    job_dir, input_dir, work_dir, out_dir, meta_path = _job_paths(job_id)
    ply = out_dir / GAUSSIANS_PLY_NAME
    if not ply.exists():
        raise HTTPException(404, detail="gaussians.ply not found")
    return FileResponse(str(ply), media_type="application/octet-stream", filename="gaussians.ply")
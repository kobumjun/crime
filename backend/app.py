# backend/app.py
from __future__ import annotations

import os
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import redis
from rq import Queue

# ============================================================
# Paths / Config
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RUN_COLMAP_SH = BASE_DIR / "recon" / "run_colmap.sh"
RUN_3DGS_SH = BASE_DIR / "recon" / "run_3dgs.sh"  # ✅ 추가: 3DGS 파이프라인(선택)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
FRONT_ORIGINS = os.environ.get("FRONT_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")

r = redis.from_url(REDIS_URL)
q = Queue("recon", connection=r)

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="CSRAI MVP (mac)", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in FRONT_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Helpers
# ============================================================
def _job_paths(job_id: str):
    job_dir = DATA_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    meta_path = job_dir / "meta.json"
    return job_dir, input_dir, output_dir, meta_path


def _read_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _write_meta(meta_path: Path, meta: Dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _tail(s: str, n: int = 6000) -> str:
    return (s or "")[-n:]


def _best_ply_path(output_dir: Path) -> Optional[Path]:
    """
    우선순위:
    1) output/points.ply (우리가 최종으로 잡는 이름)
    2) output/points_dense.ply
    3) output/points_sparse.ply
    4) output/dense/0/points.ply
    """
    candidates = [
        output_dir / "points.ply",
        output_dir / "points_dense.ply",
        output_dir / "points_sparse.ply",
        output_dir / "dense" / "0" / "points.ply",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _best_splat_path(output_dir: Path) -> Optional[Path]:
    """
    3DGS 결과 파일(.splat) 우선순위:
    1) output/scene.splat
    2) output/gaussians.splat
    3) output/3dgs/scene.splat
    """
    candidates = [
        output_dir / "scene.splat",
        output_dir / "gaussians.splat",
        output_dir / "3dgs" / "scene.splat",
        output_dir / "3dgs" / "gaussians.splat",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ============================================================
# Worker Task
# ============================================================
def recon_job(job_id: str) -> Dict[str, Any]:
    """
    파이프라인:
    A) run_colmap.sh -> sparse points 생성 (Mac CPU OK)
    B) run_3dgs.sh (있으면) -> scene.splat 생성 (대부분 CUDA 필요)
       - CUDA 없으면 실패 가능 -> done_sparse로 종료(정상)
    """
    job_dir, input_dir, output_dir, meta_path = _job_paths(job_id)
    if not job_dir.exists():
        raise RuntimeError(f"job_dir not found: {job_dir}")

    meta = _read_meta(meta_path)
    meta["status"] = "running"
    _write_meta(meta_path, meta)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not RUN_COLMAP_SH.exists():
        meta["status"] = "failed"
        meta["error"] = f"run_colmap.sh not found: {RUN_COLMAP_SH}"
        _write_meta(meta_path, meta)
        return meta

    # -------------------------
    # A) COLMAP (sparse)
    # -------------------------
    try:
        cmd = ["bash", str(RUN_COLMAP_SH), str(input_dir), str(output_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        meta["last_cmd"] = " ".join(cmd)
        meta["stdout_tail"] = _tail(proc.stdout)
        meta["stderr_tail"] = _tail(proc.stderr)

        # run_colmap.sh가 dense에서 CUDA로 죽어도, sparse 산출물이 있으면 "done_sparse"로 살린다.
        ply = _best_ply_path(output_dir)
        if proc.returncode != 0:
            if ply and ply.exists():
                # ✅ sparse라도 결과 있으면 정상 처리
                meta["status"] = "done_sparse"
                meta["warning"] = f"COLMAP pipeline returned code={proc.returncode}, but sparse output exists."
            else:
                meta["status"] = "failed"
                meta["error"] = f"COLMAP script failed (code={proc.returncode})"
                _write_meta(meta_path, meta)
                return meta

        # 결과 ply가 output/points.ply로 통일되게 맞춰주기
        ply = _best_ply_path(output_dir)
        if not ply:
            meta["status"] = "failed"
            meta["error"] = "No PLY produced (sparse/dense not found)"
            _write_meta(meta_path, meta)
            return meta

        # output/points.ply로 복사(뷰어/다운로드 일관성)
        unified = output_dir / "points.ply"
        if ply.resolve() != unified.resolve():
            shutil.copyfile(ply, unified)
            meta["points_ply_name"] = unified.name
        else:
            meta["points_ply_name"] = ply.name

        meta["points_ply"] = str(unified)

        # -------------------------
        # B) 3DGS (optional)
        # -------------------------
        # CUDA/환경 있으면 여기서 바로 학습 -> scene.splat 산출
        # 없으면 그냥 done_sparse로 끝나도 OK
        if RUN_3DGS_SH.exists():
            meta["status"] = "running_3dgs"
            _write_meta(meta_path, meta)

            cmd2 = ["bash", str(RUN_3DGS_SH), str(input_dir), str(output_dir)]
            proc2 = subprocess.run(cmd2, capture_output=True, text=True)

            meta["last_cmd_3dgs"] = " ".join(cmd2)
            meta["stdout_tail_3dgs"] = _tail(proc2.stdout)
            meta["stderr_tail_3dgs"] = _tail(proc2.stderr)

            splat = _best_splat_path(output_dir)

            if proc2.returncode == 0 and splat and splat.exists():
                # ✅ 3DGS 성공
                meta["status"] = "done_3dgs"
                meta["splat"] = str(splat)
                meta["splat_name"] = splat.name
                _write_meta(meta_path, meta)
                return meta

            # ✅ 3DGS 실패해도 sparse는 성공이면 끝까지 살린다
            meta["status"] = "done_sparse"
            meta["warning_3dgs"] = f"3DGS failed (code={proc2.returncode}). Sparse result is available."
            if "Dense stereo reconstruction requires CUDA" in (meta.get("stderr_tail_3dgs", "") + meta.get("stderr_tail", "")):
                meta["hint"] = "3DGS/Dense는 CUDA GPU 필요. 배포는 GPU 서버에서 학습/생성하도록 분리하는 게 정답."
            _write_meta(meta_path, meta)
            return meta

        # 3DGS 스크립트 없으면 sparse 완료
        if meta.get("status") not in ("done_sparse", "done_3dgs"):
            meta["status"] = "done_sparse"
        _write_meta(meta_path, meta)
        return meta

    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = f"Exception: {repr(e)}"
        _write_meta(meta_path, meta)
        return meta


# ============================================================
# Routes
# ============================================================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(files: List[UploadFile] = File(...)):
    if not files or len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 images")

    job_id = uuid.uuid4().hex[:12]
    job_dir, input_dir, output_dir, meta_path = _job_paths(job_id)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        name = _safe_filename(f.filename or "image.jpg")
        out_path = input_dir / name
        if out_path.exists():
            stem = out_path.stem
            suf = out_path.suffix
            out_path = input_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suf}"

        data = await f.read()
        out_path.write_bytes(data)
        saved.append(out_path.name)

    meta = {
        "job_id": job_id,
        "status": "queued",
        "input_count": len(saved),
        "files": saved,
    }
    _write_meta(meta_path, meta)

    rq_job = q.enqueue(recon_job, job_id)
    meta["rq_id"] = rq_job.get_id()
    _write_meta(meta_path, meta)

    return {"job_id": job_id, "status": "queued", "input_count": len(saved)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job_dir, input_dir, output_dir, meta_path = _job_paths(job_id)
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    meta = _read_meta(meta_path)

    ply = _best_ply_path(output_dir)
    splat = _best_splat_path(output_dir)

    status = meta.get("status", "unknown")

    return {
        "job_id": job_id,
        "status": status,
        "input_count": meta.get("input_count", 0),
        "files": meta.get("files", []),
        "error": meta.get("error"),
        "warning": meta.get("warning"),
        "warning_3dgs": meta.get("warning_3dgs"),
        "hint": meta.get("hint"),
        "points_ply_exists": bool(ply and ply.exists()),
        "points_ply_url": f"/api/jobs/{job_id}/points.ply" if ply and ply.exists() else None,
        "splat_exists": bool(splat and splat.exists()),
        "splat_url": f"/api/jobs/{job_id}/scene.splat" if splat and splat.exists() else None,
    }


@app.get("/api/jobs/{job_id}/points.ply")
def download_points_ply(job_id: str):
    job_dir, input_dir, output_dir, meta_path = _job_paths(job_id)
    ply = _best_ply_path(output_dir)
    if not ply or not ply.exists():
        raise HTTPException(status_code=404, detail="PLY not found (job not done?)")

    # 통일 파일명으로 내려가게
    return FileResponse(str(ply), media_type="application/octet-stream", filename="points.ply")


@app.get("/api/jobs/{job_id}/scene.splat")
def download_splat(job_id: str):
    job_dir, input_dir, output_dir, meta_path = _job_paths(job_id)
    splat = _best_splat_path(output_dir)
    if not splat or not splat.exists():
        raise HTTPException(status_code=404, detail="SPLAT not found (3DGS not done?)")

    return FileResponse(str(splat), media_type="application/octet-stream", filename="scene.splat")
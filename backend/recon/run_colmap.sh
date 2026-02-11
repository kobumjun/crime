#!/usr/bin/env bash
set -euo pipefail

# 사용법:
# ./run_colmap.sh <INPUT_DIR> <OUT_DIR>
# 예: ./run_colmap.sh /abs/path/to/input /abs/path/to/output

INPUT_DIR="${1:-}"
OUT_DIR="${2:-}"

if [[ -z "$INPUT_DIR" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 <INPUT_DIR> <OUT_DIR>"
  exit 1
fi

mkdir -p "$OUT_DIR"
DB_PATH="$OUT_DIR/colmap.db"
SPARSE_DIR="$OUT_DIR/sparse"
DENSE_DIR="$OUT_DIR/dense"

echo "[1/6] feature_extractor"
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$INPUT_DIR" \
  --ImageReader.single_camera 1

echo "[2/6] exhaustive_matcher"
colmap exhaustive_matcher \
  --database_path "$DB_PATH"

echo "[3/6] mapper (sparse reconstruction)"
mkdir -p "$SPARSE_DIR"
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$INPUT_DIR" \
  --output_path "$SPARSE_DIR"

# 0번 모델 선택 (대부분 0)
MODEL_DIR="$SPARSE_DIR/0"
if [[ ! -d "$MODEL_DIR" ]]; then
  MODEL_DIR="$(ls -d "$SPARSE_DIR"/* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: No sparse model generated."
  exit 1
fi

echo "[4/6] model_converter -> points_sparse.ply"
colmap model_converter \
  --input_path "$MODEL_DIR" \
  --output_path "$OUT_DIR/points_sparse.ply" \
  --output_type PLY

echo "[5/6] image_undistorter (prepare dense)"
mkdir -p "$DENSE_DIR"
colmap image_undistorter \
  --image_path "$INPUT_DIR" \
  --input_path "$MODEL_DIR" \
  --output_path "$DENSE_DIR" \
  --output_type COLMAP

echo "[6/6] patch_match_stereo + stereo_fusion -> points_dense.ply"
# dense 단계: 시간이 좀 걸릴 수 있음. (사진 수/해상도에 따라)
colmap patch_match_stereo \
  --workspace_path "$DENSE_DIR" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
  --workspace_path "$DENSE_DIR" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$OUT_DIR/points_dense.ply"

echo "DONE: $OUT_DIR/points_sparse.ply and $OUT_DIR/points_dense.ply"
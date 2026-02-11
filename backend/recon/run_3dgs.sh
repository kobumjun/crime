#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$1"
OUT_DIR="$2"

# 여기에 "CUDA GPU 서버에서 돌아갈 3DGS 학습 파이프라인"을 넣는다.
# 예: colmap 결과(poses/intrinsics) + images로 3DGS 학습 → scene.splat export

# 지금은 일부러 실패시키지 말고, "아직 구현 안됨"을 명확히 찍고 종료(코드=1)
echo "[3DGS] run_3dgs.sh is not implemented yet. Provide a CUDA/GPU training pipeline here."
exit 1
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="eval/official_phase1_with_b4"
mkdir -p "$OUT_DIR"

run_model() {
  local model="$1"
  local out="$2"
  shift 2

  echo "============================================================"
  echo "Running model: $model (with BasqueGLUE_bec)"
  echo "Output: $out"
  echo "============================================================"

  python3 eval/run_official_phase1.py \
    --model "$model" \
    --limit-eustrivia 80 \
    --limit-xnli 80 \
    --limit-bglue-qnli 80 \
    --enable-b4-template \
    --limit-b4-template 80 \
    --out "$out" \
    "$@"
}

run_model "kimu-2b" "$OUT_DIR/kimu-2b.json"
run_model "kimu-9b" "$OUT_DIR/kimu-9b.json"
run_model "latxa-8b" "$OUT_DIR/latxa-8b.json"
run_model "latxa-qwen3-vl-8b" "$OUT_DIR/latxa-qwen3-vl-8b.json"
run_model "latxa-70b" "$OUT_DIR/latxa-70b.json"
run_model "qwen3.5-27b" "$OUT_DIR/qwen3.5-27b-eval.json" --max-tokens 4096 --timeout 300

echo "Done. B4-inclusive single-seed batch complete."
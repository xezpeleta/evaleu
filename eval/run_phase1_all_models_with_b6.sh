#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="eval/official_phase1_with_b6"
mkdir -p "$OUT_DIR"

run_model() {
  local model="$1"
  local out="$OUT_DIR/${model}.json"
  echo "==> Running ${model} -> ${out}"

  if [[ "$model" == "qwen3.5-27b" ]]; then
    python3 eval/run_official_phase1.py \
      --model "$model" \
      --max-tokens 4096 \
      --timeout 300 \
      --limit-eustrivia 80 \
      --limit-xnli 80 \
      --limit-bglue-qnli 80 \
      --enable-b4-template --limit-b4-template 80 \
      --enable-b5-template --limit-b5-template 80 \
      --enable-b6-template --limit-b6-template 80 \
      --out "$out"
  else
    python3 eval/run_official_phase1.py \
      --model "$model" \
      --limit-eustrivia 80 \
      --limit-xnli 80 \
      --limit-bglue-qnli 80 \
      --enable-b4-template --limit-b4-template 80 \
      --enable-b5-template --limit-b5-template 80 \
      --enable-b6-template --limit-b6-template 80 \
      --out "$out"
  fi
}

run_model "kimu-2b"
run_model "kimu-9b"
run_model "latxa-8b"
run_model "latxa-70b"
run_model "qwen3.5-27b"

echo "Done. Outputs in $OUT_DIR"
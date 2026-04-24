#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="eval/official_phase1_multiseed_with_b6"
mkdir -p "$OUT_DIR"

SEEDS=(42 123 777)
MODELS=(kimu-2b kimu-9b latxa-8b latxa-qwen3-vl-4b latxa-qwen3-vl-8b latxa-qwen3-vl-32b latxa-70b qwen3.5-27b)

run_one() {
  local model="$1"
  local seed="$2"
  local out="$OUT_DIR/${model}.seed${seed}.json"
  echo "==> Running ${model} seed=${seed} -> ${out}"

  if [[ "$model" == "qwen3.5-27b" ]]; then
    python3 eval/run_official_phase1.py \
      --model "$model" \
      --seed "$seed" \
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
      --seed "$seed" \
      --limit-eustrivia 80 \
      --limit-xnli 80 \
      --limit-bglue-qnli 80 \
      --enable-b4-template --limit-b4-template 80 \
      --enable-b5-template --limit-b5-template 80 \
      --enable-b6-template --limit-b6-template 80 \
      --out "$out"
  fi
}

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_one "$model" "$seed"
  done
done

python3 eval/summarize_multiseed.py \
  --input-dir "$OUT_DIR" \
  --out "$OUT_DIR/summary.json"

echo "Done. Outputs + summary in $OUT_DIR"
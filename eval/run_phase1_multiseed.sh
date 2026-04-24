#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="eval/official_phase1_multiseed"
mkdir -p "$OUT_DIR"

SEEDS=(42 123 777)
MODELS=("kimu-2b" "kimu-9b" "latxa-8b" "latxa-qwen3-vl-8b" "latxa-70b" "qwen3.5-27b")

run_one() {
  local model="$1"
  local seed="$2"
  local out="$OUT_DIR/${model}_seed${seed}.json"

  echo "============================================================"
  echo "Running model=$model seed=$seed"
  echo "Output: $out"
  echo "============================================================"

  if [[ "$model" == "qwen3.5-27b" ]]; then
    python3 eval/run_official_phase1.py \
      --model "$model" \
      --seed "$seed" \
      --limit-eustrivia 80 \
      --limit-xnli 80 \
      --limit-bglue-qnli 80 \
      --max-tokens 4096 \
      --timeout 300 \
      --out "$out"
  else
    python3 eval/run_official_phase1.py \
      --model "$model" \
      --seed "$seed" \
      --limit-eustrivia 80 \
      --limit-xnli 80 \
      --limit-bglue-qnli 80 \
      --out "$out"
  fi
}

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_one "$model" "$seed"
  done
done

echo "============================================================"
echo "Computing multi-seed summary"
echo "============================================================"
python3 eval/summarize_multiseed.py --input-dir "$OUT_DIR" --out "$OUT_DIR/summary.json"

echo "Done. Multi-seed evaluation complete."

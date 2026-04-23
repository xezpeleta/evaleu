#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="eval/official_phase1"
mkdir -p "$OUT_DIR"

run_model() {
  local model="$1"
  local out="$2"
  shift 2

  echo "============================================================"
  echo "Running model: $model"
  echo "Output: $out"
  echo "============================================================"

  python3 eval/run_official_phase1.py \
    --model "$model" \
    --limit-eustrivia 80 \
    --limit-xnli 80 \
    --limit-bglue-qnli 80 \
    --out "$out" \
    "$@"
}

run_model "kimu-2b" "$OUT_DIR/kimu-2b.json"
run_model "kimu-9b" "$OUT_DIR/kimu-9b.json"
run_model "latxa-8b" "$OUT_DIR/latxa-8b.json"
run_model "latxa-70b" "$OUT_DIR/latxa-70b.json"
run_model "qwen3.5-27b" "$OUT_DIR/qwen3.5-27b-eval.json" --max-tokens 4096 --timeout 300

echo "============================================================"
echo "Rebuilding BasqueGLUE error report and site data"
echo "============================================================"
python3 eval/analyze_basqueglue_errors.py --results-dir "$OUT_DIR" --out "$OUT_DIR/basqueglue_error_report.json"
python3 site/build_site_data.py

echo "Done. Results + site data refreshed."

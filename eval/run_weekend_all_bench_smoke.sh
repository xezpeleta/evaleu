#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-eval/official_multiseed_allbench_smoke}"
mkdir -p "$OUT_DIR"

python3 eval/run_official_phase1.py \
  --model "${MODEL:-kimu-9b}" \
  --seed "${SEED:-42}" \
  --limit-eustrivia 2 \
  --limit-xnli 2 \
  --limit-bglue-qnli 2 \
  --enable-b4-template --limit-b4-template 2 \
  --enable-b5-template --limit-b5-template 2 \
  --enable-b6-template --limit-b6-template 2 \
  --enable-latxa-eusexams --limit-latxa-eusexams 2 \
  --enable-latxa-eusproficiency --limit-latxa-eusproficiency 2 \
  --enable-latxa-eusreading --limit-latxa-eusreading 2 \
  --out "$OUT_DIR/smoke.json"

python3 eval/summarize_multiseed.py --input-dir "$OUT_DIR" --out "$OUT_DIR/summary.json"
python3 site/build_site_data.py --summary "$OUT_DIR/summary.json" --out "site/data.json"

echo "Smoke OK: $OUT_DIR"
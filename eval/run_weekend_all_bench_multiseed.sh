#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-eval/official_multiseed_allbench_weekend}"
SUMMARY_PATH="$OUT_DIR/summary.json"
SEEDS_CSV="${SEEDS_CSV:-42,123,777}"
MODELS_CSV="${MODELS_CSV:-kimu-2b,kimu-9b,latxa-8b,latxa-qwen3-vl-8b,latxa-70b,qwen3.5-27b}"

LIMIT_EUSTRIVIA="${LIMIT_EUSTRIVIA:-80}"
LIMIT_XNLI="${LIMIT_XNLI:-80}"
LIMIT_BGLUE_QNLI="${LIMIT_BGLUE_QNLI:-80}"
LIMIT_BGLUE_BEC="${LIMIT_BGLUE_BEC:-80}"
LIMIT_BGLUE_WIC="${LIMIT_BGLUE_WIC:-80}"
LIMIT_BGLUE_INTENT="${LIMIT_BGLUE_INTENT:-80}"
LIMIT_LATXA_EUSEXAMS="${LIMIT_LATXA_EUSEXAMS:-80}"
LIMIT_LATXA_EUSPROFICIENCY="${LIMIT_LATXA_EUSPROFICIENCY:-80}"
LIMIT_LATXA_EUSREADING="${LIMIT_LATXA_EUSREADING:-80}"

mkdir -p "$OUT_DIR"

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a MODELS <<< "$MODELS_CSV"

run_one() {
  local model="$1"
  local seed="$2"
  local out="$OUT_DIR/${model}_seed${seed}.json"

  if [[ -f "$out" ]]; then
    echo "[skip] exists: $out"
    return 0
  fi

  echo "==> Running model=${model} seed=${seed}"

  local extra_args=()
  if [[ "$model" == "qwen3.5-27b" ]]; then
    extra_args+=(--max-tokens 4096 --timeout 300)
  fi

  python3 eval/run_official_phase1.py \
    --model "$model" \
    --seed "$seed" \
    --limit-eustrivia "$LIMIT_EUSTRIVIA" \
    --limit-xnli "$LIMIT_XNLI" \
    --limit-bglue-qnli "$LIMIT_BGLUE_QNLI" \
    --enable-b4-template --limit-b4-template "$LIMIT_BGLUE_BEC" \
    --enable-b5-template --limit-b5-template "$LIMIT_BGLUE_WIC" \
    --enable-b6-template --limit-b6-template "$LIMIT_BGLUE_INTENT" \
    --enable-latxa-eusexams --limit-latxa-eusexams "$LIMIT_LATXA_EUSEXAMS" \
    --enable-latxa-eusproficiency --limit-latxa-eusproficiency "$LIMIT_LATXA_EUSPROFICIENCY" \
    --enable-latxa-eusreading --limit-latxa-eusreading "$LIMIT_LATXA_EUSREADING" \
    --out "$out" \
    "${extra_args[@]}"
}

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_one "$model" "$seed"
  done
done

python3 eval/summarize_multiseed.py \
  --input-dir "$OUT_DIR" \
  --out "$SUMMARY_PATH"

python3 site/build_site_data.py --summary "$SUMMARY_PATH" --out "site/data.json"

echo "Done. Summary: $SUMMARY_PATH"
echo "Site data refreshed: site/data.json"
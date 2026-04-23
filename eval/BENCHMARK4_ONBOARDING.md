# Benchmark-4 Onboarding Template (Official Runner)

This project now includes a benchmark-registry hook in:
- `eval/run_official_phase1.py`

## Current Benchmark-4 implementation

- **Implemented benchmark**: `BasqueGLUE_bec` (sentiment-style classification)
- Labels: `N` (negative), `NEU` (neutral), `P` (positive)
- Dataset source: `orai-nlp/basqueGLUE`, config `bec`, split `test`

CLI enablement:
- `--enable-b4-template`
- `--limit-b4-template <N>`

When enabled, output JSON `limits` includes `BasqueGLUE_bec`.

## How to run with BEC (smoke)

```bash
python3 eval/run_official_phase1.py \
  --model kimu-9b \
  --limit-eustrivia 20 \
  --limit-xnli 20 \
  --limit-bglue-qnli 20 \
  --enable-b4-template \
  --limit-b4-template 20 \
  --out eval/official_phase1/with_bec_smoke.json
```

## How to swap Benchmark-4 later

If you want a different Benchmark-4 in the future:
1) Edit `build_benchmark4_template_items()`
2) Keep output item schema:

```python
{
  "bench": "<benchmark_id>",
  "id": "unique_id",
  "prompt": "...",
  "gold": 0,
  "label_names": ["label_a", "label_b"],
  "meta": {}
}
```

3) Update `build_benchmark_registry()` benchmark id if changed
4) Add bench-specific parsing branch in `score_item()` if needed

## Notes

- Keep endpoint private (`LLAMA_SWAP_BASE_URL` via `.env`).
- qwen no-thinking control remains active in chat payload.
- Registry-based `limits` auto-reflect enabled benchmarks.

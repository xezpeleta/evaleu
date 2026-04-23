# Basque LLM Evaluation (llama-swap, multi-seed robust)

This repository benchmarks Basque-capable local LLMs served behind a private `llama-swap` endpoint and publishes a static comparison website.

The current website view is **robust multi-seed** (mean performance across seeds), not single-run-only.

---

## What is evaluated

### Models (current set)
- `latxa-70b`
- `qwen3.5-27b` (evaluated in no-thinking mode)
- `kimu-9b`
- `latxa-8b`
- `kimu-2b`

### Benchmarks (current set)
1. **EusTrivia** — 4-way multiple-choice factual/cultural QA in Basque
2. **XNLIeu** — Basque NLI (`entailment / neutral / contradiction`)
3. **BasqueGLUE-QNLI** — Basque sentence-pair inference (`entailment / not_entailment`)
4. **BasqueGLUE-BEC** — Basque sentiment classification (`N / NEU / P`)

---

## Methodology

### 1) Deterministic sampling and decoding
- `temperature = 0`
- Fixed seed runs
- Equal sampling budget per benchmark

### 2) Evaluation budget
- **80 items per benchmark**
- **4 benchmarks**
- **320 items/model/seed**

### 3) Multi-seed robustness
Current leaderboard is computed across seeds:
- `42`, `123`, `777`

For each model and benchmark we report:
- **mean accuracy** across seeds (used for ranking)
- **std** across seeds (shown on hover in UI)

### 4) Parsing/scoring rules (important)
- Exact-match style classification scoring.
- BasqueGLUE-QNLI parser avoids the classic collision where `entailment` is incorrectly matched inside `not_entailment`.
- Label matching uses normalized and ordered checks to reduce ambiguous parses.

### 5) Qwen stability mode
For `qwen3.5-27b`, evaluation disables thinking/reasoning mode and uses higher token/timeout settings to avoid empty-content responses.

### 6) Website presentation policy
- Table cells show simple rounded values (e.g. `72.5%`).
- Hover tooltip shows full robust value (e.g. `72.5% ± 1.9%`).
- Best score per metric column is shown in **bold**.

---

## Repository structure

- `eval/run_official_phase1.py` — core evaluator (registry-based benchmark hooks)
- `eval/run_phase1_all_models.sh` — 3-benchmark single-seed all-model batch
- `eval/run_phase1_multiseed.sh` — 3-benchmark multi-seed batch
- `eval/run_phase1_all_models_with_b4.sh` — 4-benchmark single-seed all-model batch
- `eval/run_phase1_multiseed_with_b4.sh` — 4-benchmark multi-seed batch
- `eval/summarize_multiseed.py` — aggregates per-model mean/std from per-seed JSONs
- `eval/analyze_basqueglue_errors.py` — BasqueGLUE diagnostics
- `site/model_cards.json` — canonical model metadata (family, quantization, release date, source model card URL)
- `site/build_site_data.py` — builds `site/data.json` from robust summary + model cards
- `site/index.html` — static report UI

---

## Reproduce current website data (4 benchmarks, multi-seed)

From repo root:

```bash
./eval/run_phase1_multiseed_with_b4.sh
python3 site/build_site_data.py
python3 -m http.server 8787
```

Open:
- `http://127.0.0.1:8787/site/`

---

## Privacy and endpoint hygiene

- The endpoint is private and must not be committed in tracked files.
- Use `.env` for local endpoint configuration (`LLAMA_SWAP_BASE_URL`).
- Published site artifacts keep endpoint as placeholder (`${LLAMA_SWAP_BASE_URL}`).

---

## Notes

- If browser cache shows stale data, open with a cache-busting query, e.g. `?v=timestamp`.
- Prefer adding new benchmarks only after multi-seed stability checks on the current suite.
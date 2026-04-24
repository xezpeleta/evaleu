# Basque LLM Evaluation (llama-swap, multi-seed robust)

This repository benchmarks Basque-capable local LLMs served behind a private `llama-swap` endpoint and publishes a static comparison website.

The website is a **robust multi-seed** view (mean across seeds; std on hover), and now supports grouped benchmark families to keep the table compact.

---

## Models (current set)
- `latxa-70b`
- `qwen3.5-27b` (evaluated in no-thinking mode)
- `kimu-9b`
- `latxa-8b`
- `kimu-2b`

---

## Benchmarks (all-bench suite)

### Core
1. **EusTrivia**
2. **XNLIeu**

### BasqueGLUE
3. **BasqueGLUE-QNLI**
4. **BasqueGLUE-BEC**
5. **BasqueGLUE-WiC**
6. **BasqueGLUE-Intent**

### LatxaEvalSuite
7. **LatxaEval-EusExams**
8. **LatxaEval-EusProficiency**
9. **LatxaEval-EusReading**

---

## Methodology

- `temperature=0`
- Fixed seed runs (`42, 123, 777` by default)
- Equal sampling budget per benchmark (default `80` items)
- Ranking by **mean overall accuracy** across seeds
- Hover/tooltip displays **mean ± std**
- Best score per column highlighted in **bold**

For `qwen3.5-27b`, evaluation uses no-thinking mode with higher token/timeout settings for stability.

---

## Repository structure

- `eval/run_official_phase1.py` — core evaluator (registry-based benchmark hooks)
- `eval/summarize_multiseed.py` — aggregates per-model mean/std from per-seed JSONs
- `eval/run_weekend_all_bench_smoke.sh` — 1-model quick smoke for all 9 benchmarks
- `eval/run_weekend_all_bench_multiseed.sh` — full all-model multi-seed weekend runner
- `site/model_cards.json` — canonical model metadata (family, quantization, release date, source URL)
- `site/build_site_data.py` — builds `site/data.json` from robust summary + model cards
- `site/index.html` — static report UI (family-grouped columns)
- `docs/weekend-server-runbook.md` — server execution runbook

---

## Quickstart

### 1) Smoke test (recommended)

```bash
./eval/run_weekend_all_bench_smoke.sh
```

### 2) Full weekend run

```bash
./eval/run_weekend_all_bench_multiseed.sh
```

### 3) Build/serve site

```bash
python3 site/build_site_data.py
python3 -m http.server 8787
```

Open: `http://127.0.0.1:8787/site/`

---

## Privacy and endpoint hygiene

- Keep endpoint in local `.env` (`LLAMA_SWAP_BASE_URL=...`)
- Do not commit private URLs in tracked files/artifacts
- Published artifacts should keep endpoint placeholder (`${LLAMA_SWAP_BASE_URL}`)
- Runbook includes pre-push privacy grep

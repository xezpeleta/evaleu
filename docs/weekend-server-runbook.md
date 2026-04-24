# Weekend all-benchmark runbook (server host)

This runbook executes a full multi-seed run with grouped benchmark families:
- Core: EusTrivia, XNLIeu
- BasqueGLUE: QNLI, BEC, WiC, Intent
- LatxaEvalSuite: EusExams, EusProficiency, EusReading

## 0) Prerequisites

- Python env with project deps installed.
- Access to private llama-swap endpoint from server.
- `.env` present on server (not committed):

```bash
LLAMA_SWAP_BASE_URL=http://<private-host>:8080
```

## 1) Sync repo and sanity check

```bash
git pull --ff-only
python3 -m py_compile eval/run_official_phase1.py eval/summarize_multiseed.py site/build_site_data.py
bash -n eval/run_weekend_all_bench_multiseed.sh eval/run_weekend_all_bench_smoke.sh
```

## 2) Fast smoke test (recommended)

```bash
./eval/run_weekend_all_bench_smoke.sh
```

Expected artifacts:
- `eval/official_multiseed_allbench_smoke/smoke.json`
- `eval/official_multiseed_allbench_smoke/summary.json`
- `site/data.json`

## 3) Launch weekend full run

Default run: 5 models × 3 seeds × 9 benchmarks × 80 items.

```bash
./eval/run_weekend_all_bench_multiseed.sh
```

### Useful overrides

```bash
# custom output directory
OUT_DIR=eval/official_multiseed_allbench_weekend_r2 ./eval/run_weekend_all_bench_multiseed.sh

# custom seeds/models
SEEDS_CSV=42,123,777,2026 MODELS_CSV=kimu-9b,latxa-8b ./eval/run_weekend_all_bench_multiseed.sh

# reduce load (example)
LIMIT_EUSTRIVIA=40 LIMIT_XNLI=40 LIMIT_BGLUE_QNLI=40 \
LIMIT_BGLUE_BEC=40 LIMIT_BGLUE_WIC=40 LIMIT_BGLUE_INTENT=40 \
LIMIT_LATXA_EUSEXAMS=40 LIMIT_LATXA_EUSPROFICIENCY=40 LIMIT_LATXA_EUSREADING=40 \
./eval/run_weekend_all_bench_multiseed.sh
```

## 4) Outputs

Full script writes:
- Per-run JSONs: `eval/official_multiseed_allbench_weekend/*.json`
- Aggregated summary: `eval/official_multiseed_allbench_weekend/summary.json`
- Website payload: `site/data.json`

## 5) Publish web (when ready)

```bash
git add site/data.json
git commit -m "chore(site): refresh data from weekend all-benchmark multiseed run"
git push origin main

# publish static site branch
git subtree split --prefix site -b gh-pages
git push -u origin gh-pages --force
```

## 6) Privacy checks (mandatory)

Before commit/push:

```bash
python3 - << 'PY'
import pathlib,re
root=pathlib.Path('.')
pat=re.compile(r'studio\.tknika\.net|http://studio|https://studio',re.I)
for p in root.rglob('*'):
    if p.is_file() and '.git' not in p.parts and p.suffix in {'.py','.sh','.md','.json','.html','.yml','.yaml'}:
        try:
            t=p.read_text(encoding='utf-8',errors='ignore')
        except Exception:
            continue
        if pat.search(t):
            print('FOUND',p)
PY
```

No private endpoint URL should appear in tracked files.

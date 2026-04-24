# Eval Runbook (single CLI)

This project is operated through `uv run evaleu.py`.

## 1) Preflight
```bash
python3 -m py_compile evaleu.py eval/run_eval.py eval/summarize_multiseed.py site/build_site_data.py
python3 -m json.tool site/model_cards.json >/dev/null
```

## 2) Add model metadata (if new model)
```bash
uv run evaleu.py model \
  --id <model-id> \
  --display-name "<Display Name>" \
  --family "<Family>" \
  --params "<Params>" \
  --upstream-model-id <org/model> \
  --release-date-utc <ISO8601> \
  --release-source-url <URL>
```

## 3) Evaluate one model (default)
```bash
uv run evaleu.py eval --model <model-id>
```
Outputs:
- per-seed json: `eval/<model-id>_seed<seed>.json`
- summary: `eval/summary.json`
- website data: `site/data.json`

## 4) Status while running
```bash
uv run evaleu.py status
```

## 5) Rebuild only
```bash
uv run evaleu.py summarize
uv run evaleu.py build
```

## 6) Cleanup legacy artifacts
```bash
uv run evaleu.py clean --apply
```

## 7) Publish
```bash
git add -A
git commit -m "data: refresh eval summary and site data"
git push origin main
```
GitHub Actions deploys `site/` automatically on push.

## Privacy
- Keep endpoint in local `.env`
- Never commit private endpoint URLs

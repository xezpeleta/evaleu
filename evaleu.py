#!/usr/bin/env python3
"""Unified CLI for Basque LLM evaluation workflow.

Examples:
  uv run evaleu.py eval --model latxa-qwen3-vl-4b
  uv run evaleu.py eval-all
  uv run evaleu.py summarize
  uv run evaleu.py build
  uv run evaleu.py model-add --id my-model --display-name "My Model" --family "Llama" --params "8B" \
    --upstream-model-id org/model --release-date-utc 2026-01-01T00:00:00Z --release-source-url https://huggingface.co/org/model
  uv run evaleu.py status
  uv run evaleu.py clean --apply

Backwards compatibility:
  uv run evaleu.py --model latxa-qwen3-vl-4b
  (interpreted as: uv run evaleu.py eval --model ...)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_CARDS_PATH = ROOT / "site" / "model_cards.json"

LEGACY_PATHS = [
    "eval/run_phase1_all_models.sh",
    "eval/run_phase1_all_models_with_b4.sh",
    "eval/run_phase1_all_models_with_b5.sh",
    "eval/run_phase1_all_models_with_b6.sh",
    "eval/run_phase1_multiseed.sh",
    "eval/run_phase1_multiseed_with_b4.sh",
    "eval/run_phase1_multiseed_with_b5.sh",
    "eval/run_phase1_multiseed_with_b6.sh",
    "eval/run_weekend_all_bench_multiseed.sh",
    "eval/run_weekend_all_bench_smoke.sh",
    "eval/report_weekend_status.py",
    "docs/weekend-server-runbook.md",
    "eval/official_phase1",
    "eval/official_phase1_with_b4",
    "eval/official_phase1_with_b5",
    "eval/official_phase1_multiseed",
    "eval/official_phase1_multiseed_with_b4",
    "eval/official_phase1_multiseed_with_b5",
    "eval/official_multiseed_allbench_smoke",
    "eval/official_multiseed_allbench_tmp",
    "eval/official_multiseed_allbench_weekend",
]


def run_cmd(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    # Avoid inheriting uv-run isolation values into child Python executions.
    for key in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def default_python() -> str:
    return "/usr/bin/python3" if Path("/usr/bin/python3").exists() else "python3"


def parse_csv(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def load_model_cards() -> dict:
    return json.loads(MODEL_CARDS_PATH.read_text(encoding="utf-8"))


def save_model_cards(cards: dict) -> None:
    MODEL_CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_ids_from_cards() -> list[str]:
    return list(load_model_cards().keys())


def summary_from_out_dir(out_dir: Path, explicit_summary: str | None) -> Path:
    if explicit_summary:
        p = Path(explicit_summary)
        return p if p.is_absolute() else (ROOT / p).resolve()
    return (out_dir / "summary.json").resolve()


def add_common_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seeds", default="42,123,777", help="Comma-separated seeds")
    p.add_argument("--out-dir", default="eval", help="Directory for per-seed outputs + summary.json")
    p.add_argument("--summary", default=None, help="Summary output path (default: <out-dir>/summary.json)")
    p.add_argument("--site-data", default="site/data.json", help="Site payload output path")
    p.add_argument("--force", action="store_true", help="Re-run even if output json already exists")
    p.add_argument("--python", default=default_python(), help="Python interpreter for underlying scripts")
    p.add_argument("--no-summarize", action="store_true", help="Do not run summarize step")
    p.add_argument("--no-build", action="store_true", help="Do not run site build step")

    p.add_argument("--limit-eustrivia", type=int, default=80)
    p.add_argument("--limit-xnli", type=int, default=80)
    p.add_argument("--limit-bglue-qnli", type=int, default=80)
    p.add_argument("--limit-bglue-bec", type=int, default=80)
    p.add_argument("--limit-bglue-wic", type=int, default=80)
    p.add_argument("--limit-bglue-intent", type=int, default=80)
    p.add_argument("--limit-latxa-eusexams", type=int, default=80)
    p.add_argument("--limit-latxa-eusproficiency", type=int, default=80)
    p.add_argument("--limit-latxa-eusreading", type=int, default=80)


def run_one_model_eval(args: argparse.Namespace, model_id: str) -> None:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in parse_csv(args.seeds):
        out_file = out_dir / f"{model_id}_seed{seed}.json"
        if out_file.exists() and not args.force:
            print(f"[skip] exists: {out_file}")
            continue

        cmd = [
            args.python,
            "eval/run_eval.py",
            "--model",
            model_id,
            "--seed",
            seed,
            "--limit-eustrivia",
            str(args.limit_eustrivia),
            "--limit-xnli",
            str(args.limit_xnli),
            "--limit-bglue-qnli",
            str(args.limit_bglue_qnli),
            "--enable-b4-template",
            "--limit-b4-template",
            str(args.limit_bglue_bec),
            "--enable-b5-template",
            "--limit-b5-template",
            str(args.limit_bglue_wic),
            "--enable-b6-template",
            "--limit-b6-template",
            str(args.limit_bglue_intent),
            "--enable-latxa-eusexams",
            "--limit-latxa-eusexams",
            str(args.limit_latxa_eusexams),
            "--enable-latxa-eusproficiency",
            "--limit-latxa-eusproficiency",
            str(args.limit_latxa_eusproficiency),
            "--enable-latxa-eusreading",
            "--limit-latxa-eusreading",
            str(args.limit_latxa_eusreading),
            "--out",
            str(out_file),
        ]

        if model_id == "qwen3.5-27b":
            cmd += ["--max-tokens", "4096", "--timeout", "300"]

        run_cmd(cmd)


def run_summarize(python: str, out_dir: str, summary: str | None) -> Path:
    out = Path(out_dir)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    summ = summary_from_out_dir(out, summary)
    run_cmd([python, "eval/summarize_multiseed.py", "--input-dir", str(out), "--out", str(summ)])
    return summ


def run_build(python: str, summary: str, site_data: str) -> None:
    summary_p = Path(summary)
    if not summary_p.is_absolute():
        summary_p = (ROOT / summary).resolve()
    site_data_p = Path(site_data)
    if not site_data_p.is_absolute():
        site_data_p = (ROOT / site_data).resolve()
    run_cmd([python, "site/build_site_data.py", "--summary", str(summary_p), "--out", str(site_data_p)])


def cmd_eval(args: argparse.Namespace) -> int:
    run_one_model_eval(args, args.model)

    summary_p = summary_from_out_dir((ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir), args.summary)
    summarized = False
    built = False
    if not args.no_summarize:
        summary_p = run_summarize(args.python, args.out_dir, args.summary)
        summarized = True
    if not args.no_build:
        run_build(args.python, str(summary_p), args.site_data)
        built = True

    if summarized:
        print(f"Summary refreshed: {summary_p}")
    else:
        print(f"Summary skipped. Current path: {summary_p}")

    if built:
        print(f"Site data refreshed: {args.site_data}")
    else:
        print("Site build skipped.")
    return 0


def cmd_eval_all(args: argparse.Namespace) -> int:
    models = parse_csv(args.models_csv) if args.models_csv else model_ids_from_cards()
    if not models:
        raise SystemExit("No models provided and no model_cards entries found")
    for model_id in models:
        run_one_model_eval(args, model_id)

    summary_p = summary_from_out_dir((ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir), args.summary)
    summarized = False
    built = False
    if not args.no_summarize:
        summary_p = run_summarize(args.python, args.out_dir, args.summary)
        summarized = True
    if not args.no_build:
        run_build(args.python, str(summary_p), args.site_data)
        built = True

    if summarized:
        print(f"Summary refreshed: {summary_p}")
    else:
        print(f"Summary skipped. Current path: {summary_p}")

    if built:
        print(f"Site data refreshed: {args.site_data}")
    else:
        print("Site build skipped.")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    summary = run_summarize(args.python, args.out_dir, args.summary)
    print(f"Summary refreshed: {summary}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    run_build(args.python, args.summary, args.site_data)
    print(f"Site data refreshed: {args.site_data}")
    return 0


def cmd_model_add(args: argparse.Namespace) -> int:
    cards = load_model_cards()
    if args.id in cards and not args.force:
        raise SystemExit(f"Model '{args.id}' already exists. Use --force to overwrite.")

    cards[args.id] = {
        "display_name": args.display_name,
        "family": args.family,
        "params": args.params,
        "weights_quant": args.weights_quant,
        "kv_cache": args.kv_cache,
        "upstream_model_id": args.upstream_model_id,
        "release_date_utc": args.release_date_utc,
        "release_source_url": args.release_source_url,
    }
    save_model_cards(cards)
    print(f"Added/updated model card: {args.id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    models = parse_csv(args.models_csv) if args.models_csv else model_ids_from_cards()
    seeds = parse_csv(args.seeds)
    expected = len(models) * len(seeds)

    run_files = sorted(p for p in out_dir.glob("*_seed*.json") if p.name != "summary.json")
    done = len(run_files)
    pct = round((100.0 * done / expected), 1) if expected else 0.0

    latest_mtime = None
    if run_files:
        latest = max(run_files, key=lambda p: p.stat().st_mtime)
        latest_mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).isoformat()

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "expected_runs": expected,
        "completed_runs": done,
        "progress_percent": pct,
        "summary_exists": (out_dir / "summary.json").exists(),
        "latest_result_mtime_utc": latest_mtime,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    root = ROOT
    removed: list[str] = []
    for rel in LEGACY_PATHS:
        p = root / rel
        if not p.exists():
            continue
        if args.apply:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        removed.append(rel)

    print(json.dumps({
        "mode": "apply" if args.apply else "dry_run",
        "removed_or_planned": removed,
        "count": len(removed),
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Unified CLI for eval, summarize, build, model registry, and cleanup")
    sub = ap.add_subparsers(dest="cmd")

    p_eval = sub.add_parser("eval", help="Evaluate one model, then summarize and build by default")
    p_eval.add_argument("--model", required=True)
    add_common_eval_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_eval_all = sub.add_parser("eval-all", help="Evaluate all models from model_cards (or --models-csv)")
    p_eval_all.add_argument("--models-csv", default=None, help="Comma-separated model IDs (default: all in model_cards)")
    add_common_eval_args(p_eval_all)
    p_eval_all.set_defaults(func=cmd_eval_all)

    p_sum = sub.add_parser("summarize", help="Generate eval/summary.json from per-seed eval/*.json")
    p_sum.add_argument("--out-dir", default="eval")
    p_sum.add_argument("--summary", default=None, help="Default: <out-dir>/summary.json")
    p_sum.add_argument("--python", default=default_python())
    p_sum.set_defaults(func=cmd_summarize)

    p_build = sub.add_parser("build", help="Build site/data.json from summary")
    p_build.add_argument("--summary", default="eval/summary.json")
    p_build.add_argument("--site-data", default="site/data.json")
    p_build.add_argument("--python", default=default_python())
    p_build.set_defaults(func=cmd_build)

    p_model = sub.add_parser("model-add", help="Add or update a model card entry")
    p_model.add_argument("--id", required=True)
    p_model.add_argument("--display-name", required=True)
    p_model.add_argument("--family", required=True)
    p_model.add_argument("--params", required=True)
    p_model.add_argument("--upstream-model-id", required=True)
    p_model.add_argument("--release-date-utc", required=True)
    p_model.add_argument("--release-source-url", required=True)
    p_model.add_argument("--weights-quant", default="F16")
    p_model.add_argument("--kv-cache", default="f16/f16")
    p_model.add_argument("--force", action="store_true")
    p_model.set_defaults(func=cmd_model_add)

    p_status = sub.add_parser("status", help="Show progress based on eval/<model>_seed<seed>.json files")
    p_status.add_argument("--out-dir", default="eval")
    p_status.add_argument("--models-csv", default=None, help="Comma-separated expected models (default: model_cards)")
    p_status.add_argument("--seeds", default="42,123,777")
    p_status.set_defaults(func=cmd_status)

    p_clean = sub.add_parser("clean", help="Remove legacy scripts and legacy eval folders")
    p_clean.add_argument("--apply", action="store_true", help="Actually delete files; default is dry-run")
    p_clean.set_defaults(func=cmd_clean)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or os.sys.argv[1:])

    # Back-compat: if user calls `uv run evaleu.py --model ...`, treat as `eval`.
    # Keep top-level help/version-like flags as top-level behavior.
    if argv and argv[0] in {"--model", "-m"}:
        argv = ["eval", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

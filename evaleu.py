#!/usr/bin/env python3
"""Run one-model multiseed Basque benchmark eval, then rebuild site payload.

Example:
  uv run evaleu.py --model latxa-qwen3-vl-8b
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    # Avoid inheriting uv-run Python isolation into child processes.
    for k in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(k, None)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate one model (multi-seed, all benchmarks), summarize, and rebuild site/data.json."
    )
    ap.add_argument("--model", required=True, help="Model id served by llama-swap (e.g. latxa-qwen3-vl-8b)")
    ap.add_argument("--seeds", default="42,123,777", help="Comma-separated seeds (default: 42,123,777)")
    ap.add_argument("--out-dir", default="eval/official_multiseed_allbench_weekend", help="Directory for per-seed json outputs")
    ap.add_argument("--summary", default=None, help="Summary output path (default: <out-dir>/summary.json)")
    ap.add_argument("--site-data", default="site/data.json", help="Site payload output path")
    ap.add_argument("--force", action="store_true", help="Re-run even if per-seed output json already exists")

    ap.add_argument("--limit-eustrivia", type=int, default=80)
    ap.add_argument("--limit-xnli", type=int, default=80)
    ap.add_argument("--limit-bglue-qnli", type=int, default=80)
    ap.add_argument("--limit-bglue-bec", type=int, default=80)
    ap.add_argument("--limit-bglue-wic", type=int, default=80)
    ap.add_argument("--limit-bglue-intent", type=int, default=80)
    ap.add_argument("--limit-latxa-eusexams", type=int, default=80)
    ap.add_argument("--limit-latxa-eusproficiency", type=int, default=80)
    ap.add_argument("--limit-latxa-eusreading", type=int, default=80)

    default_python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else "python3"
    ap.add_argument("--python", default=default_python, help="Python interpreter to use for underlying scripts")

    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else out_dir / "summary.json"
    if not summary_path.is_absolute():
        summary_path = (root / summary_path).resolve()
    site_data_path = Path(args.site_data)
    if not site_data_path.is_absolute():
        site_data_path = (root / site_data_path).resolve()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No seeds provided")

    for seed in seeds:
        out_file = out_dir / f"{args.model}_seed{seed}.json"
        if out_file.exists() and not args.force:
            print(f"[skip] exists: {out_file}")
            continue

        cmd = [
            args.python,
            "eval/run_official_phase1.py",
            "--model",
            args.model,
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

        if args.model == "qwen3.5-27b":
            cmd += ["--max-tokens", "4096", "--timeout", "300"]

        run(cmd, cwd=root)

    run(
        [
            args.python,
            "eval/summarize_multiseed.py",
            "--input-dir",
            str(out_dir),
            "--out",
            str(summary_path),
        ],
        cwd=root,
    )

    run(
        [
            args.python,
            "site/build_site_data.py",
            "--summary",
            str(summary_path),
            "--out",
            str(site_data_path),
        ],
        cwd=root,
    )

    print(f"Done. Summary: {summary_path}")
    print(f"Site data refreshed: {site_data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

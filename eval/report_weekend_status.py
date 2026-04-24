#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_csv(v: str):
    return [x.strip() for x in v.split(',') if x.strip()]


def tail_lines(path: Path, n: int = 12):
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
        return lines[-n:]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description='Report progress of weekend all-benchmark run')
    ap.add_argument('--out-dir', default='eval/official_multiseed_allbench_weekend')
    ap.add_argument('--models-csv', default='kimu-2b,kimu-9b,latxa-8b,latxa-qwen3-vl-4b,latxa-qwen3-vl-8b,latxa-qwen3-vl-32b,latxa-70b,qwen3.5-27b')
    ap.add_argument('--seeds-csv', default='42,123,777')
    ap.add_argument('--tail', type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    models = parse_csv(args.models_csv)
    seeds = parse_csv(args.seeds_csv)
    expected = len(models) * len(seeds)

    run_files = sorted(out_dir.glob('*_seed*.json'))
    run_files = [p for p in run_files if p.name != 'summary.json']
    done = len(run_files)
    pct = (100.0 * done / expected) if expected else 0.0

    summary = out_dir / 'summary.json'
    log_path = out_dir / 'run.log'

    latest_mtime = None
    if run_files:
        latest = max(run_files, key=lambda p: p.stat().st_mtime)
        latest_mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).isoformat()

    tail = tail_lines(log_path, args.tail)
    current = None
    for line in reversed(tail):
        if '==> Running model=' in line:
            current = line.strip()
            break

    payload = {
        'timestamp_utc': now_iso(),
        'out_dir': str(out_dir),
        'expected_runs': expected,
        'completed_runs': done,
        'progress_percent': round(pct, 1),
        'summary_exists': summary.exists(),
        'log_exists': log_path.exists(),
        'latest_result_mtime_utc': latest_mtime,
        'current_hint': current,
        'tail': tail,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

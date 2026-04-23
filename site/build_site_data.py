#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

def load_model_cards(root: Path):
    model_cards_path = root / "site" / "model_cards.json"
    return json.loads(model_cards_path.read_text(encoding="utf-8"))

ALL_BENCHMARKS = [
    {
        "id": "EusTrivia",
        "task": "Multiple-choice factual and cultural knowledge in Basque",
        "metric": "Accuracy",
        "labels": "4 options (A/B/C/D)",
    },
    {
        "id": "XNLIeu",
        "task": "Natural language inference in Basque (premise-hypothesis)",
        "metric": "Accuracy",
        "labels": "entailment / neutral / contradiction",
    },
    {
        "id": "BasqueGLUE_qnli",
        "task": "Question-Answer NLI from BasqueGLUE (sentence pair classification)",
        "metric": "Accuracy",
        "labels": "entailment / not_entailment",
    },
    {
        "id": "BasqueGLUE_bec",
        "task": "Sentiment classification from BasqueGLUE BEC",
        "metric": "Accuracy",
        "labels": "N / NEU / P",
    },
    {
        "id": "BasqueGLUE_wic",
        "task": "Word-in-Context disambiguation from BasqueGLUE WiC",
        "metric": "Accuracy",
        "labels": "false / true",
    },
]


def main():
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "eval" / "official_phase1_multiseed_with_b5" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_cards = load_model_cards(root)

    models_map = summary.get("models", {})
    present_benchmark_ids = {
        b
        for s in models_map.values()
        for b in (s.get("benchmarks", {}) or {}).keys()
    }
    benchmark_ids = [b["id"] for b in ALL_BENCHMARKS if b["id"] in present_benchmark_ids]

    rows = []
    all_seed_ids = set()

    for model_id, s in models_map.items():
        meta = model_cards.get(model_id, {
            "display_name": model_id,
            "family": "unknown",
            "params": "unknown",
            "weights_quant": "unknown",
            "kv_cache": "unknown",
            "upstream_model_id": "unknown",
            "release_date_utc": None,
            "release_source_url": None,
        })

        runs = s.get("runs", [])
        for r in runs:
            if r.get("seed") is not None:
                all_seed_ids.add(int(r["seed"]))

        bench_means = s.get("benchmarks", {})
        by_benchmark = {}
        for b in benchmark_ids:
            bm = bench_means.get(b, {})
            by_benchmark[b] = {
                "n": 80,
                "accuracy": bm.get("mean", 0.0),
                "accuracy_std": bm.get("std", 0.0),
                "coverage": 1.0,
            }

        rows.append({
            "model_id": model_id,
            "display_name": meta["display_name"],
            "family": meta["family"],
            "params": meta["params"],
            "weights_quant": meta["weights_quant"],
            "kv_cache": meta["kv_cache"],
            "upstream_model_id": meta.get("upstream_model_id"),
            "release_date_utc": meta.get("release_date_utc"),
            "release_source_url": meta.get("release_source_url"),
            "overall_accuracy": s.get("overall_mean", 0.0),
            "overall_accuracy_std": s.get("overall_std", 0.0),
            "n_items": 80 * len(by_benchmark),
            "n_seeds": len(runs),
            "seed_ids": sorted([r.get("seed") for r in runs if r.get("seed") is not None]),
            "by_benchmark": by_benchmark,
            "source_file": str(summary_path.relative_to(root)),
        })

    rows.sort(key=lambda x: x["overall_accuracy"], reverse=True)

    benchmark_defs = [b for b in ALL_BENCHMARKS if b["id"] in benchmark_ids]
    benchmark_label_list = ", ".join([
        {
            "BasqueGLUE_qnli": "BasqueGLUE-QNLI",
            "BasqueGLUE_bec": "BasqueGLUE-BEC",
            "BasqueGLUE_wic": "BasqueGLUE-WiC",
        }.get(b["id"], b["id"])
        for b in benchmark_defs
    ])
    n_items_per_model = 80 * len(benchmark_defs)

    out = {
        "title": "Basque LLM Evaluation",
        "subtitle": f"Comparative evaluation on {benchmark_label_list} (multi-seed robust view)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "${LLAMA_SWAP_BASE_URL}",
        "evaluation_protocol": {
            "suite": "Official benchmark subset (multi-seed)",
            "metric": "Accuracy",
            "sampling": f"80 items per benchmark ({n_items_per_model} total per model)",
            "decoding": "temperature=0",
            "seeds": sorted(all_seed_ids),
            "benchmarks": benchmark_defs,
        },
        "results": rows,
    }

    out_path = root / "site" / "data.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)

if __name__ == "__main__":
    main()

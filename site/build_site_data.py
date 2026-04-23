#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

MODEL_META = {
    "latxa-70b": {
        "display_name": "Latxa 70B",
        "family": "Llama 3.1",
        "params": "70B",
        "weights_quant": "Q8_0",
        "kv_cache": "default",
    },
    "qwen3.5-27b-eval": {
        "display_name": "Qwen 3.5 27B (no-thinking)",
        "family": "Qwen 3.5",
        "params": "27B",
        "weights_quant": "Q8_0",
        "kv_cache": "bf16/bf16",
    },
    "qwen3.5-27b": {
        "display_name": "Qwen 3.5 27B (no-thinking)",
        "family": "Qwen 3.5",
        "params": "27B",
        "weights_quant": "Q8_0",
        "kv_cache": "bf16/bf16",
    },
    "kimu-9b": {
        "display_name": "Kimu 9B",
        "family": "Gemma-Kimu",
        "params": "9B",
        "weights_quant": "Q8_0",
        "kv_cache": "default",
    },
    "latxa-8b": {
        "display_name": "Latxa 8B",
        "family": "Llama 3.1",
        "params": "8B",
        "weights_quant": "F16",
        "kv_cache": "f16/f16",
    },
    "kimu-2b": {
        "display_name": "Kimu 2B",
        "family": "Gemma-Kimu",
        "params": "2B",
        "weights_quant": "Q8_0",
        "kv_cache": "default",
    },
}

BENCHMARKS = [
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
]
BENCHMARK_IDS = [b["id"] for b in BENCHMARKS]


def main():
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "eval" / "official_phase1_multiseed_with_b4" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    rows = []
    all_seed_ids = set()

    for model_id, s in summary.get("models", {}).items():
        meta = MODEL_META.get(model_id, {
            "display_name": model_id,
            "family": "unknown",
            "params": "unknown",
            "weights_quant": "unknown",
            "kv_cache": "unknown",
        })

        runs = s.get("runs", [])
        for r in runs:
            if r.get("seed") is not None:
                all_seed_ids.add(int(r["seed"]))

        bench_means = s.get("benchmarks", {})
        by_benchmark = {}
        for b in BENCHMARK_IDS:
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
            "overall_accuracy": s.get("overall_mean", 0.0),
            "overall_accuracy_std": s.get("overall_std", 0.0),
            "n_items": 320,
            "n_seeds": len(runs),
            "seed_ids": sorted([r.get("seed") for r in runs if r.get("seed") is not None]),
            "by_benchmark": by_benchmark,
            "source_file": str(summary_path.relative_to(root)),
        })

    rows.sort(key=lambda x: x["overall_accuracy"], reverse=True)

    out = {
        "title": "Basque LLM Evaluation",
        "subtitle": "Comparative evaluation on EusTrivia, XNLIeu, BasqueGLUE-QNLI, and BasqueGLUE-BEC (multi-seed robust view)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "${LLAMA_SWAP_BASE_URL}",
        "evaluation_protocol": {
            "suite": "Official benchmark subset (multi-seed)",
            "metric": "Accuracy",
            "sampling": "80 items per benchmark (320 total per model)",
            "decoding": "temperature=0",
            "seeds": sorted(all_seed_ids),
            "benchmarks": BENCHMARKS,
        },
        "results": rows,
    }

    out_path = root / "site" / "data.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)

if __name__ == "__main__":
    main()

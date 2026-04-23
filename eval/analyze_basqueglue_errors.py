#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any


def load_label_names() -> List[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("orai-nlp/basqueGLUE", "qnli", trust_remote_code=True)["test"]
        return list(ds.features["label"].names)
    except Exception:
        return ["entailment", "not_entailment"]


def read_results(results_dir: Path) -> Dict[str, Any]:
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        model = data.get("model", p.stem)
        items = [x for x in data.get("items", []) if x.get("bench") == "BasqueGLUE_qnli"]
        out[model] = {"path": str(p), "items": items, "summary": data.get("by_benchmark", {}).get("BasqueGLUE_qnli", {})}
    return out


def analyze_model(items: List[Dict[str, Any]], label_names: List[str]) -> Dict[str, Any]:
    n = len(items)
    correct = sum(1 for x in items if x.get("ok") is True)
    coverage = sum(1 for x in items if x.get("pred_label") is not None)

    conf = defaultdict(int)
    pred_dist = Counter()
    gold_dist = Counter()
    answer_dist = Counter()
    parse_fail = []
    wrong_examples = []

    for x in items:
        g = x.get("gold_label")
        p = x.get("pred_label")
        a = (x.get("answer") or "").strip()
        conf[(g, p)] += 1
        gold_dist[g] += 1
        pred_dist[p] += 1
        answer_dist[a] += 1

        if p is None:
            parse_fail.append({"id": x.get("id"), "answer": a})

        if x.get("ok") is False:
            wrong_examples.append({
                "id": x.get("id"),
                "gold": g,
                "pred": p,
                "answer": a,
            })

    labels = list(range(len(label_names)))
    matrix = []
    for g in labels:
        row = []
        for p in labels:
            row.append(conf[(g, p)])
        row.append(conf[(g, None)])
        matrix.append(row)

    return {
        "n": n,
        "accuracy": (correct / n) if n else 0.0,
        "coverage": (coverage / n) if n else 0.0,
        "confusion_matrix": {
            "rows_gold": label_names,
            "cols_pred": label_names + ["UNPARSED"],
            "values": matrix,
        },
        "gold_distribution": {str(k): v for k, v in gold_dist.items()},
        "pred_distribution": {str(k): v for k, v in pred_dist.items()},
        "top_answers": answer_dist.most_common(15),
        "parse_failures": parse_fail,
        "wrong_examples": wrong_examples[:20],
    }


def pretty_print(model: str, rep: Dict[str, Any], label_names: List[str]) -> None:
    print(f"\n=== {model} ===")
    print(f"N={rep['n']}  accuracy={rep['accuracy']:.4f}  coverage={rep['coverage']:.4f}")
    print("gold_distribution:", rep["gold_distribution"])
    print("pred_distribution:", rep["pred_distribution"])

    cm = rep["confusion_matrix"]
    cols = cm["cols_pred"]
    print("confusion_matrix (gold x pred):")
    print("  cols:", cols)
    for i, row in enumerate(cm["values"]):
        print(f"  gold={label_names[i]:<15} -> {row}")

    print("top_answers:")
    for ans, c in rep["top_answers"][:10]:
        print(f"  {c:>3}  {ans}")

    if rep["parse_failures"]:
        print(f"parse_failures: {len(rep['parse_failures'])}")
        for ex in rep["parse_failures"][:8]:
            print("  ", ex)
    else:
        print("parse_failures: 0")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze BasqueGLUE_qnli errors from official phase1 result JSON files")
    ap.add_argument("--results-dir", default="eval/official_phase1", help="Directory with per-model JSON results")
    ap.add_argument("--out", default="eval/official_phase1/basqueglue_error_report.json", help="Output JSON report path")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    label_names = load_label_names()
    model_data = read_results(results_dir)

    report = {
        "benchmark": "BasqueGLUE_qnli",
        "label_names": label_names,
        "models": {},
    }

    for model, payload in model_data.items():
        rep = analyze_model(payload["items"], label_names)
        rep["source_file"] = payload["path"]
        rep["summary_accuracy"] = payload["summary"].get("accuracy")
        rep["summary_coverage"] = payload["summary"].get("coverage")
        report["models"][model] = rep
        pretty_print(model, rep, label_names)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()

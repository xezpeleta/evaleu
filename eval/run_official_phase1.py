#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests
from datasets import load_dataset


def _load_dotenv(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _resolve_base_url(cli_base_url: str | None) -> str:
    if cli_base_url:
        return cli_base_url
    return os.environ.get("LLAMA_SWAP_BASE_URL", "http://127.0.0.1:8080")


def _max_tokens_for_model(model: str, cli_max_tokens: int | None) -> int:
    if cli_max_tokens is not None:
        return cli_max_tokens
    return 3072 if model == "qwen3.5-27b" else 256


def _timeout_for_model(model: str, cli_timeout: int | None) -> int:
    if cli_timeout is not None:
        return cli_timeout
    return 240 if model == "qwen3.5-27b" else 120


def _extract_answer(msg: Dict[str, Any]) -> str:
    content = (msg.get("content") or "").strip()
    if content:
        return content
    reasoning = (msg.get("reasoning_content") or "").strip()
    if not reasoning:
        return ""

    lines = [ln.strip() for ln in reasoning.splitlines() if ln.strip()]
    for ln in lines:
        if ln.lower().startswith("final:"):
            return ln.split(":", 1)[1].strip()
    markers = ["final answer", "answer:", "therefore", "thus", "so the answer is"]
    for ln in reversed(lines):
        low = ln.lower()
        if any(m in low for m in markers):
            return ln.split(":", 1)[1].strip() if ":" in ln else ln
    return lines[-1] if lines else ""


def chat_completion(base_url: str, model: str, prompt: str, temperature: float, max_tokens: int, timeout: int, retries: int = 2) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Erantzun bakarra eman. Ez azaldu arrazoiketa."
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Qwen reasoning control for llama-server chat template
    if model.startswith("qwen"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            finish = data["choices"][0].get("finish_reason")

            # Qwen deepseek reasoning can return empty content on length stop.
            # Retry once with higher max_tokens before falling back to reasoning extraction.
            if (not (msg.get("content") or "").strip()) and finish == "length" and max_tokens < 8192:
                payload2 = dict(payload)
                payload2["max_tokens"] = min(8192, max_tokens * 2)
                r2 = requests.post(url, json=payload2, timeout=timeout)
                r2.raise_for_status()
                data2 = r2.json()
                msg2 = data2["choices"][0]["message"]
                return _extract_answer(msg2).strip()

            return _extract_answer(msg).strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise last_err


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_choice_letter(text: str) -> str | None:
    t = _normalize(text)
    m = re.search(r"\b([abcd])\b", t)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([1-4])\b", t)
    if m:
        return chr(ord('A') + int(m.group(1)) - 1)
    return None


def _label_from_text(answer: str, names: List[str]) -> int | None:
    t = _normalize(answer)
    if t.isdigit():
        idx = int(t)
        if 0 <= idx < len(names):
            return idx

    # Prefer exact/word-boundary matches and check longer labels first
    # (e.g., "not_entailment" contains "entailment").
    normalized = [(_normalize(n), i) for i, n in enumerate(names)]
    normalized.sort(key=lambda x: len(x[0]), reverse=True)

    # exact whole answer
    for name, idx in normalized:
        if t == name:
            return idx

    # token/word-boundary variants (underscore/hyphen/space interchangeable)
    t_soft = re.sub(r"[_\-]", " ", t)
    for name, idx in normalized:
        n_soft = re.sub(r"[_\-]", " ", name)
        if re.search(rf"\b{re.escape(n_soft)}\b", t_soft):
            return idx

    return None


def build_eustrivia_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("HiTZ/EusTrivia")["test"]
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        cands = row["candidates"]
        letters = ["A", "B", "C", "D"]
        opts = "\n".join(f"{letters[j]}) {cands[j]}" for j in range(len(cands)))
        prompt = (
            "Aukeratu aukera zuzena (A/B/C/D bakarrik).\n"
            f"Galdera: {row['question']}\n{opts}\n"
            "Erantzuna:"
        )
        items.append({
            "bench": "EusTrivia",
            "id": f"eustrivia_{row['id']}",
            "prompt": prompt,
            "gold": int(row["answer"]),
            "label_names": letters,
            "meta": {
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
            },
        })
    return items


def build_xnli_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("HiTZ/xnli-eu", "eu_native")["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    label_map = {
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
    }

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Sailkatu hipotesiaren eta premisaren arteko erlazioa. "
            "Erantzun hitz bakarrarekin: entailment, neutral, edo contradiction.\n"
            f"Premisa: {row['premise']}\n"
            f"Hipotesia: {row['hypothesis']}\n"
            "Erantzuna:"
        )
        items.append({
            "bench": "XNLIeu",
            "id": f"xnli_{i}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": [label_map[n] for n in names],
            "meta": {},
        })
    return items


def build_basqueglue_qnli_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("orai-nlp/basqueGLUE", "qnli", trust_remote_code=True)["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Sailkatu ondorengo bikotea. Erantzun ETIKETA BAKARRAREKIN, besterik ez: entailment edo not_entailment.\n"
            "Ez eman azalpenik. Ez erabili beste hitzik.\n"
            f"Galdera: {row['question']}\n"
            f"Esaldia: {row['sentence']}\n"
            "Erantzuna (entailment/not_entailment):"
        )
        items.append({
            "bench": "BasqueGLUE_qnli",
            "id": f"bg_qnli_{row['idx']}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": list(names),
            "meta": {},
        })
    return items


def build_benchmark4_template_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    """
    Benchmark-4 implementation (sentiment-style): BasqueGLUE BEC
    Labels: N (negative), NEU (neutral), P (positive)
    """
    ds = load_dataset("orai-nlp/basqueGLUE", "bec", trust_remote_code=True)["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Sailkatu testuaren sentimendua. Erantzun ETIKETA BAKARRAREKIN, besterik ez: N, NEU, edo P.\n"
            "Ez eman azalpenik. Ez erabili beste hitzik.\n"
            f"Testua: {row['text']}\n"
            "Erantzuna (N/NEU/P):"
        )
        items.append({
            "bench": "BasqueGLUE_bec",
            "id": f"bg_bec_{row['idx']}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": list(names),
            "meta": {},
        })
    return items


def build_benchmark5_template_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    """
    Benchmark-5 implementation (lexical semantics): BasqueGLUE WiC
    Labels: false / true
    """
    ds = load_dataset("orai-nlp/basqueGLUE", "wic", trust_remote_code=True)["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Esan hitzak bi esaldietan esanahi bera duen ala ez. "
            "Erantzun ETIKETA BAKARRAREKIN: true edo false.\n"
            "Ez eman azalpenik. Ez erabili beste hitzik.\n"
            f"Hitza: {row['word']}\n"
            f"Esaldia 1: {row['sentence1']}\n"
            f"Esaldia 2: {row['sentence2']}\n"
            "Erantzuna (true/false):"
        )
        items.append({
            "bench": "BasqueGLUE_wic",
            "id": f"bg_wic_{row['idx']}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": list(names),
            "meta": {"word": row.get("word")},
        })
    return items


def build_benchmark6_template_items(limit: int, seed: int) -> List[Dict[str, Any]]:
    """
    Benchmark-6 implementation (intent classification): BasqueGLUE Intent
    Labels: 12 intent classes
    """
    ds = load_dataset("orai-nlp/basqueGLUE", "intent", trust_remote_code=True)["test"]
    names = ds.features["label"].names
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:limit]

    label_block = "\n".join([f"{i}: {name}" for i, name in enumerate(names)])

    items = []
    for i in idxs:
        row = ds[int(i)]
        prompt = (
            "Sailkatu erabiltzailearen asmoa (intent).\n"
            "Aukeratu zerrendako etiketa zuzena eta erantzun ZENBAKI BAKARRAREKIN (0-11), besterik ez.\n"
            "Ez eman azalpenik. Ez erabili etiketa testurik.\n"
            f"Etiketak:\n{label_block}\n\n"
            f"Testua: {row['text']}\n"
            "Erantzuna (0-11):"
        )
        items.append({
            "bench": "BasqueGLUE_intent",
            "id": f"bg_intent_{row['idx']}",
            "prompt": prompt,
            "gold": int(row["label"]),
            "label_names": list(names),
            "meta": {},
        })
    return items


def _postprocess_eustrivia_candidates(items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    ds_eu = load_dataset("HiTZ/EusTrivia")["test"]
    eu_by_id = {f"eustrivia_{r['id']}": r for r in ds_eu}
    for it in items:
        row = eu_by_id.get(it["id"])
        if row:
            it.setdefault("meta", {})["candidates"] = row["candidates"]


def build_benchmark_registry(args: argparse.Namespace) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "id": "EusTrivia",
            "limit": args.limit_eustrivia,
            "builder": build_eustrivia_items,
            "postprocess": _postprocess_eustrivia_candidates,
        },
        {
            "id": "XNLIeu",
            "limit": args.limit_xnli,
            "builder": build_xnli_items,
        },
        {
            "id": "BasqueGLUE_qnli",
            "limit": args.limit_bglue_qnli,
            "builder": build_basqueglue_qnli_items,
        },
    ]

    if args.enable_b4_template or args.limit_b4_template > 0:
        specs.append(
            {
                "id": "BasqueGLUE_bec",
                "limit": args.limit_b4_template,
                "builder": build_benchmark4_template_items,
            }
        )

    if args.enable_b5_template or args.limit_b5_template > 0:
        specs.append(
            {
                "id": "BasqueGLUE_wic",
                "limit": args.limit_b5_template,
                "builder": build_benchmark5_template_items,
            }
        )

    if args.enable_b6_template or args.limit_b6_template > 0:
        specs.append(
            {
                "id": "BasqueGLUE_intent",
                "limit": args.limit_b6_template,
                "builder": build_benchmark6_template_items,
            }
        )

    return specs


def build_items_from_registry(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    items: List[Dict[str, Any]] = []
    limits: Dict[str, int] = {}

    for spec in build_benchmark_registry(args):
        limit = int(spec.get("limit", 0) or 0)
        bench_id = spec["id"]
        limits[bench_id] = limit
        if limit <= 0:
            continue

        built = spec["builder"](limit, args.seed)
        post = spec.get("postprocess")
        if post:
            post(built)
        items.extend(built)

    return items, limits


@dataclass
class Pred:
    bench: str
    item_id: str
    answer: str
    pred_label: int | None
    gold_label: int
    ok: bool


def score_item(item: Dict[str, Any], answer: str) -> Tuple[int | None, bool]:
    label_names = item["label_names"]
    pred = None

    if item["bench"] == "EusTrivia":
        letter = _extract_choice_letter(answer)
        if letter:
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(label_names):
                pred = idx
        if pred is None:
            t = _normalize(answer)
            for i, cand in enumerate(item.get("meta", {}).get("candidates", []) or []):
                if _normalize(cand) in t:
                    pred = i
                    break
    elif item["bench"] == "BasqueGLUE_bec":
        t = _normalize(answer)
        if re.search(r"\bneu(tral)?\b", t):
            pred = 1
        elif re.search(r"\bn(egatiboa|egative)?\b", t):
            pred = 0
        elif re.search(r"\bp(ositiboa|ositive)?\b", t):
            pred = 2
        else:
            pred = _label_from_text(answer, label_names)
    elif item["bench"] == "BasqueGLUE_wic":
        t = _normalize(answer)
        if re.search(r"\b(true|berdin(a)?|bai|same)\b", t):
            pred = 1
        elif re.search(r"\b(false|desberdin(a)?|ezberdin(a)?|different)\b", t):
            pred = 0
        else:
            pred = _label_from_text(answer, label_names)
    elif item["bench"] == "BasqueGLUE_intent":
        m = re.search(r"\b(\d{1,2})\b", answer)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(label_names):
                pred = idx
        if pred is None:
            pred = _label_from_text(answer, label_names)
    else:
        pred = _label_from_text(answer, label_names)

    ok = pred == int(item["gold"]) if pred is not None else False
    return pred, ok


def aggregate(preds: List[Pred]) -> Dict[str, Any]:
    by_bench: Dict[str, List[Pred]] = {}
    for p in preds:
        by_bench.setdefault(p.bench, []).append(p)

    metrics = {}
    total = len(preds)
    total_ok = sum(1 for p in preds if p.ok)
    for b, ps in by_bench.items():
        metrics[b] = {
            "n": len(ps),
            "accuracy": sum(1 for p in ps if p.ok) / max(1, len(ps)),
            "coverage": sum(1 for p in ps if p.pred_label is not None) / max(1, len(ps)),
        }

    return {
        "overall_accuracy": total_ok / max(1, total),
        "n_items": total,
        "by_benchmark": metrics,
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root)

    ap = argparse.ArgumentParser(description="Official Phase-1 Basque benchmark runner")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=None, help="request timeout seconds (per call)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-eustrivia", type=int, default=100)
    ap.add_argument("--limit-xnli", type=int, default=100)
    ap.add_argument("--limit-bglue-qnli", type=int, default=100)
    ap.add_argument("--enable-b4-template", action="store_true", help="Enable Benchmark-4 onboarding template hook")
    ap.add_argument("--limit-b4-template", type=int, default=0, help="Sample limit for Benchmark-4 template")
    ap.add_argument("--enable-b5-template", action="store_true", help="Enable Benchmark-5 onboarding template hook")
    ap.add_argument("--limit-b5-template", type=int, default=0, help="Sample limit for Benchmark-5 template")
    ap.add_argument("--enable-b6-template", action="store_true", help="Enable Benchmark-6 onboarding template hook")
    ap.add_argument("--limit-b6-template", type=int, default=0, help="Sample limit for Benchmark-6 template")
    ap.add_argument("--out", default="eval/official_phase1/results.json")
    args = ap.parse_args()

    base_url = _resolve_base_url(args.base_url)
    max_tokens = _max_tokens_for_model(args.model, args.max_tokens)
    timeout = _timeout_for_model(args.model, args.timeout)

    items, limits = build_items_from_registry(args)

    preds: List[Pred] = []
    for i, it in enumerate(items, 1):
        ans = chat_completion(base_url, args.model, it["prompt"], args.temperature, max_tokens, timeout=timeout)
        pred_label, ok = score_item(it, ans)
        preds.append(Pred(it["bench"], it["id"], ans, pred_label, int(it["gold"]), ok))
        print(f"[{i}/{len(items)}] {it['bench']} {it['id']}: {'OK' if ok else 'FAIL'} | ans={ans!r}")

    summary = aggregate(preds)
    out = {
        "base_url": "${LLAMA_SWAP_BASE_URL}",
        "model": args.model,
        "suite": "official_phase1",
        "max_tokens": max_tokens,
        "timeout": timeout,
        "limits": limits,
        **summary,
        "items": [
            {
                "bench": p.bench,
                "id": p.item_id,
                "answer": p.answer,
                "pred_label": p.pred_label,
                "gold_label": p.gold_label,
                "ok": p.ok,
            }
            for p in preds
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print("Model:", args.model)
    print("Items:", out["n_items"])
    print("Overall accuracy:", f"{out['overall_accuracy']:.3f}")
    for b, m in out["by_benchmark"].items():
        print(f"- {b}: acc={m['accuracy']:.3f} cov={m['coverage']:.3f} n={m['n']}")
    print("Saved:", str(out_path))


if __name__ == "__main__":
    main()

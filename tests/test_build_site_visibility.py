import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "build_site_data", Path(__file__).resolve().parents[1] / "site" / "build_site_data.py"
)
build_site_data = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(build_site_data)


def _write_summary(path: Path) -> None:
    payload = {
        "models": {
            "kimu-9b": {
                "overall_mean": 0.61,
                "overall_std": 0.02,
                "runs": [{"seed": 42}],
                "benchmarks": {"EusTrivia": {"mean": 0.61, "std": 0.02}},
            },
            "hidden-9b": {
                "overall_mean": 0.71,
                "overall_std": 0.02,
                "runs": [{"seed": 42}],
                "benchmarks": {"EusTrivia": {"mean": 0.71, "std": 0.02}},
            },
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_site_excludes_draft_models_from_results(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    (fake_repo / "site").mkdir(parents=True)
    (fake_repo / "eval").mkdir(parents=True)

    model_cards = {
        "kimu-9b": {
            "display_name": "Kimu 9B",
            "family": "Kimu",
            "params": "9B",
            "weights_quant": "Q8_0",
            "kv_cache": "f16/f16",
            "upstream_model_id": "org/kimu-9b",
            "release_date_utc": "2026-01-01T00:00:00Z",
            "release_source_url": "https://huggingface.co/org/kimu-9b",
            "site_visibility": "published",
        },
        "hidden-9b": {
            "display_name": "Hidden 9B",
            "family": "Hidden",
            "params": "9B",
            "weights_quant": "Q8_0",
            "kv_cache": "f16/f16",
            "upstream_model_id": "org/hidden-9b",
            "release_date_utc": "2026-01-01T00:00:00Z",
            "release_source_url": "https://huggingface.co/org/hidden-9b",
            "site_visibility": "draft",
        },
    }
    (fake_repo / "site" / "model_cards.json").write_text(
        json.dumps(model_cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary_path = fake_repo / "eval" / "summary.json"
    out_path = fake_repo / "site" / "data.json"
    _write_summary(summary_path)

    monkeypatch.setattr(build_site_data, "__file__", str(fake_repo / "site" / "build_site_data.py"))
    monkeypatch.setattr(sys, "argv", ["build_site_data.py", "--summary", str(summary_path), "--out", str(out_path)])

    build_site_data.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    model_ids = [r["model_id"] for r in payload["results"]]
    assert "kimu-9b" in model_ids
    assert "hidden-9b" not in model_ids

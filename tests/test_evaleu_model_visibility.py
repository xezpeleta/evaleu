import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("evaleu", Path(__file__).resolve().parents[1] / "evaleu.py")
evaleu = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(evaleu)


def _base_model_args():
    return [
        "model",
        "--id",
        "kimu-9b",
        "--display-name",
        "Kimu 9B",
        "--family",
        "Kimu",
        "--params",
        "9B",
        "--upstream-model-id",
        "org/kimu-9b",
        "--release-date-utc",
        "2026-01-01T00:00:00Z",
        "--release-source-url",
        "https://huggingface.co/org/kimu-9b",
    ]


def test_model_add_defaults_to_published_visibility(tmp_path, monkeypatch):
    cards_path = tmp_path / "model_cards.json"
    cards_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaleu, "MODEL_CARDS_PATH", cards_path)

    parser = evaleu.build_parser()
    args = parser.parse_args(_base_model_args())

    rc = evaleu.cmd_model(args)
    assert rc == 0

    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    assert cards["kimu-9b"]["site_visibility"] == "published"


def test_model_add_hide_sets_draft_visibility(tmp_path, monkeypatch):
    cards_path = tmp_path / "model_cards.json"
    cards_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaleu, "MODEL_CARDS_PATH", cards_path)

    parser = evaleu.build_parser()
    args = parser.parse_args([*_base_model_args(), "--hide"])

    rc = evaleu.cmd_model(args)
    assert rc == 0

    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    assert cards["kimu-9b"]["site_visibility"] == "draft"


def test_model_visibility_toggle_preserves_existing_fields(tmp_path, monkeypatch):
    cards_path = tmp_path / "model_cards.json"
    cards_path.write_text(
        json.dumps(
            {
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
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(evaleu, "MODEL_CARDS_PATH", cards_path)

    parser = evaleu.build_parser()
    args = parser.parse_args(["model", "--id", "kimu-9b", "--force", "--hide"])

    rc = evaleu.cmd_model(args)
    assert rc == 0

    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    assert cards["kimu-9b"]["site_visibility"] == "draft"
    assert cards["kimu-9b"]["family"] == "Kimu"
    assert cards["kimu-9b"]["weights_quant"] == "Q8_0"

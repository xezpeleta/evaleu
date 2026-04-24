import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("evaleu", Path(__file__).resolve().parents[1] / "evaleu.py")
evaleu = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(evaleu)


def test_model_add_appends_no_thinking_suffix(tmp_path, monkeypatch):
    cards_path = tmp_path / "model_cards.json"
    cards_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaleu, "MODEL_CARDS_PATH", cards_path)

    parser = evaleu.build_parser()
    args = parser.parse_args(
        [
            "model",
            "--id",
            "qwen3.6-27b",
            "--display-name",
            "Qwen 3.6 27B",
            "--family",
            "Qwen 3.6",
            "--params",
            "27B",
            "--upstream-model-id",
            "Qwen/Qwen3.6-27B",
            "--release-date-utc",
            "2026-04-01T00:00:00Z",
            "--release-source-url",
            "https://huggingface.co/Qwen/Qwen3.6-27B",
            "--no-thinking",
        ]
    )

    rc = evaleu.cmd_model(args)
    assert rc == 0

    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    assert cards["qwen3.6-27b"]["display_name"] == "Qwen 3.6 27B (no-thinking)"


def test_model_add_does_not_duplicate_no_thinking_suffix(tmp_path, monkeypatch):
    cards_path = tmp_path / "model_cards.json"
    cards_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaleu, "MODEL_CARDS_PATH", cards_path)

    parser = evaleu.build_parser()
    args = parser.parse_args(
        [
            "model",
            "--id",
            "qwen3.6-27b",
            "--display-name",
            "Qwen 3.6 27B (no-thinking)",
            "--family",
            "Qwen 3.6",
            "--params",
            "27B",
            "--upstream-model-id",
            "Qwen/Qwen3.6-27B",
            "--release-date-utc",
            "2026-04-01T00:00:00Z",
            "--release-source-url",
            "https://huggingface.co/Qwen/Qwen3.6-27B",
            "--no-thinking",
        ]
    )

    rc = evaleu.cmd_model(args)
    assert rc == 0

    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    assert cards["qwen3.6-27b"]["display_name"] == "Qwen 3.6 27B (no-thinking)"

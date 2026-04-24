import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("evaleu", Path(__file__).resolve().parents[1] / "evaleu.py")
evaleu = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(evaleu)


def _capture_first_cmd(monkeypatch):
    calls = []

    def _fake_run_cmd(cmd, cwd=evaleu.ROOT):
        calls.append(cmd)

    monkeypatch.setattr(evaleu, "run_cmd", _fake_run_cmd)
    return calls


def test_eval_passes_disable_thinking_flag_to_runner(monkeypatch, tmp_path):
    parser = evaleu.build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--model",
            "kimu-9b",
            "--seeds",
            "42",
            "--out-dir",
            str(tmp_path),
            "--disable-thinking",
            "--no-summarize",
            "--no-build",
        ]
    )

    calls = _capture_first_cmd(monkeypatch)
    evaleu.run_one_model_eval(args, "kimu-9b")

    assert calls, "expected run_cmd to be called"
    assert "--disable-thinking" in calls[0]


def test_eval_does_not_pass_disable_thinking_when_not_set(monkeypatch, tmp_path):
    parser = evaleu.build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--model",
            "kimu-9b",
            "--seeds",
            "42",
            "--out-dir",
            str(tmp_path),
            "--no-summarize",
            "--no-build",
        ]
    )

    calls = _capture_first_cmd(monkeypatch)
    evaleu.run_one_model_eval(args, "kimu-9b")

    assert calls, "expected run_cmd to be called"
    assert "--disable-thinking" not in calls[0]

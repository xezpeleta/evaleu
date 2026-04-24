import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("evaleu", Path(__file__).resolve().parents[1] / "evaleu.py")
evalue = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(evalue)


def test_server_subcommand_invokes_http_server(monkeypatch, tmp_path):
    calls = []

    def _fake_run_cmd(cmd, cwd=evalue.ROOT):
        calls.append((cmd, cwd))

    monkeypatch.setattr(evalue, "run_cmd", _fake_run_cmd)

    parser = evalue.build_parser()
    args = parser.parse_args(
        [
            "server",
            "--dir",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9090",
            "--python",
            "python3",
        ]
    )

    rc = evalue.cmd_server(args)

    assert rc == 0
    assert calls == [(["python3", "-m", "http.server", "9090", "--bind", "0.0.0.0"], tmp_path)]


def test_server_subcommand_fails_for_missing_dir(tmp_path):
    parser = evalue.build_parser()
    missing = tmp_path / "does-not-exist"
    args = parser.parse_args(["server", "--dir", str(missing)])

    with pytest.raises(SystemExit, match="Server directory does not exist"):
        evalue.cmd_server(args)

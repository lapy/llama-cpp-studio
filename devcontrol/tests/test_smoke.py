from pathlib import Path


def test_tmp_path_is_writable(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("ok")
    assert (tmp_path / "x.txt").read_text() == "ok"

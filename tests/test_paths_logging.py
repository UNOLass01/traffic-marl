import logging

import pytest

from trafficmarl.utils import paths
from trafficmarl.utils.logging_utils import get_logger, setup_logging


def test_project_root_detected():
    assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()
    assert paths.CONFIGS_DIR.is_dir()


def test_resolve_relative_and_absolute(tmp_path):
    assert paths.resolve("configs/base.yaml") == paths.PROJECT_ROOT / "configs/base.yaml"
    assert paths.resolve(tmp_path) == tmp_path


def test_run_dir_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    run = paths.run_dir("grid3x3", "ippo", 0)
    assert run == tmp_path / "grid3x3" / "ippo" / "0"
    assert run.is_dir()


def test_sumo_home_required(monkeypatch):
    monkeypatch.delenv("SUMO_HOME", raising=False)
    with pytest.raises(OSError):
        paths.sumo_home()
    monkeypatch.setenv("SUMO_HOME", "/opt/sumo")
    assert paths.sumo_tools_dir() == paths.Path("/opt/sumo/tools")


def test_logging_writes_file_without_duplicate_handlers(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    setup_logging("INFO", log_file)
    logger = setup_logging("INFO", log_file)  # second call must not duplicate output
    assert len(logger.handlers) == 2
    get_logger("tests").info("hello")
    for handler in logger.handlers:
        handler.flush()
    assert log_file.read_text().count("hello") == 1


def test_get_logger_is_namespaced():
    assert get_logger("envs").name == "trafficmarl.envs"
    assert get_logger("trafficmarl.envs").name == "trafficmarl.envs"
    assert isinstance(get_logger("x"), logging.Logger)


def test_gzip_output_is_reproducible_and_readable(tmp_path):
    for name in ("a.xml.gz", "b.xml.gz"):
        with paths.open_output(tmp_path / name) as fh:
            fh.write(b"<routes/>")
    assert (tmp_path / "a.xml.gz").read_bytes() == (tmp_path / "b.xml.gz").read_bytes()
    with paths.open_input(tmp_path / "a.xml.gz") as fh:
        assert fh.read() == b"<routes/>"

import subprocess
import sys

from trafficmarl import __version__
from trafficmarl.cli import main


def test_help_via_module():
    result = subprocess.run(
        [sys.executable, "-m", "trafficmarl", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: trafficmarl" in result.stdout


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out


def test_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert "pettingzoo" in out


def test_config_command_applies_overrides(capsys):
    assert main(["config", "configs/base.yaml", "experiment.name=demo"]) == 0
    assert "name: demo" in capsys.readouterr().out


def test_config_command_reports_errors(capsys):
    assert main(["config", "configs/does_not_exist.yaml"]) == 2
    assert "not found" in capsys.readouterr().err

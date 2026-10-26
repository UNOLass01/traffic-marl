from pathlib import Path

import pytest

from trafficmarl.utils import paths
from trafficmarl.utils.config import (
    ConfigError,
    apply_overrides,
    deep_merge,
    get_by_path,
    load_config,
    parse_override,
    save_config,
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_deep_merge_merges_nested_and_replaces_lists():
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2]}
    merged = deep_merge(base, {"a": {"y": 3}, "l": [9]})
    assert merged == {"a": {"x": 1, "y": 3}, "l": [9]}
    assert base == {"a": {"x": 1, "y": 2}, "l": [1, 2]}, "inputs must not be mutated"


def test_defaults_are_merged_in_order_and_file_wins(tmp_path):
    write(tmp_path / "a.yaml", "env: {delta_time: 5, yellow: 2}\nname: a\n")
    write(tmp_path / "b.yaml", "env: {delta_time: 10}\nname: b\n")
    exp = write(tmp_path / "exp.yaml", "defaults: [a.yaml, b.yaml]\nname: exp\n")
    config = load_config(exp)
    assert config == {"env": {"delta_time": 10, "yellow": 2}, "name": "exp"}


def test_defaults_fall_back_to_configs_dir():
    config = load_config(paths.CONFIGS_DIR / "base.yaml")
    assert len(config["experiment"]["seeds"]) >= 5


def test_circular_defaults_rejected(tmp_path):
    write(tmp_path / "a.yaml", "defaults: [b.yaml]\n")
    write(tmp_path / "b.yaml", "defaults: [a.yaml]\n")
    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_file_and_non_mapping_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path / "list.yaml", "- 1\n- 2\n"))


def test_empty_file_is_empty_config(tmp_path):
    assert load_config(write(tmp_path / "empty.yaml", "")) == {}


@pytest.mark.parametrize(
    ("raw", "keys", "value"),
    [
        ("a.b=3", ["a", "b"], 3),
        ("flag=true", ["flag"], True),
        ("xs=[1, 2]", ["xs"], [1, 2]),
        ("name=grid=3x3", ["name"], "grid=3x3"),
    ],
)
def test_parse_override_types(raw, keys, value):
    assert parse_override(raw) == (keys, value)


def test_overrides_create_and_replace_keys(tmp_path):
    exp = write(tmp_path / "exp.yaml", "env: {delta_time: 5}\n")
    config = load_config(exp, ["env.delta_time=10", "reward.alpha=0.5"])
    assert config == {"env": {"delta_time": 10}, "reward": {"alpha": 0.5}}


def test_override_through_scalar_rejected():
    with pytest.raises(ConfigError):
        apply_overrides({"a": 1}, ["a.b=2"])
    with pytest.raises(ConfigError):
        parse_override("no_equals_sign")


def test_get_by_path():
    config = {"a": {"b": {"c": 1}}}
    assert get_by_path(config, "a.b.c") == 1
    assert get_by_path(config, "a.x", default="d") == "d"


def test_save_config_round_trip(tmp_path):
    config = {"b": 1, "a": {"nested": [1, 2]}}
    saved = save_config(config, tmp_path / "run" / "config.yaml")
    assert load_config(saved) == config

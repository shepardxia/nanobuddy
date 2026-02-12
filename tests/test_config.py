"""Tests for typed dataclass config with YAML + dot-override."""

import pytest
import yaml

from nanobuddy.train.config import Config, load_config


class TestLoadConfig:
    def test_defaults_from_empty_yaml(self, tmp_path):
        (tmp_path / "cfg.yaml").write_text("{}")
        cfg = load_config(tmp_path / "cfg.yaml")
        assert cfg.model.architecture == "e_branchformer"
        assert cfg.training.steps == 20000

    def test_missing_file_uses_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nope.yaml")
        assert cfg.training.steps == 20000

    def test_yaml_override(self, tmp_path):
        (tmp_path / "cfg.yaml").write_text(yaml.dump({
            "model": {"architecture": "lstm"},
            "training": {"steps": 5000},
        }))
        cfg = load_config(tmp_path / "cfg.yaml")
        assert cfg.model.architecture == "lstm"
        assert cfg.training.steps == 5000

    def test_dot_overrides(self, tmp_path):
        (tmp_path / "cfg.yaml").write_text("{}")
        cfg = load_config(tmp_path / "cfg.yaml", overrides=[
            "training.steps=30000",
            "model.architecture=gru",
            "training.loss_bias=0.8",
        ])
        assert cfg.training.steps == 30000
        assert cfg.model.architecture == "gru"
        assert cfg.training.loss_bias == pytest.approx(0.8)

    def test_extra_model_params(self, tmp_path):
        (tmp_path / "cfg.yaml").write_text(yaml.dump({
            "model": {"architecture": "tcn", "n_blocks": 3, "kernel_size": 5},
        }))
        cfg = load_config(tmp_path / "cfg.yaml")
        assert cfg.model.extra["n_blocks"] == 3
        assert cfg.model.extra["kernel_size"] == 5

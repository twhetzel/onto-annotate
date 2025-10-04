import pytest
from pathlib import Path
from onto_annotate.cli import load_config

def test_load_config_happy_path(tmp_path):
    """Checks that a valid YAML config file with keys 'ontologies' and 'columns_to_annotate' loads correctly."""
    p = tmp_path / "cfg.yml"
    p.write_text("ontologies: ['MONDO']\ncolumns_to_annotate: ['condition']\n")
    cfg = load_config(str(p))
    assert cfg["ontologies"] == ["MONDO"]
    assert cfg["columns_to_annotate"] == ["condition"]

def test_load_config_missing_keys_raises(tmp_path):
    """Checks that missing required keys cause a SystemExit."""
    p = tmp_path / "bad.yml"
    p.write_text("ontologies: ['MONDO']\n")
    with pytest.raises(SystemExit):
        load_config(str(p))

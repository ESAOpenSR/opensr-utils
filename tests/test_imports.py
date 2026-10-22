"""Import contracts must be tested before any ML imports in a fresh interpreter."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "opensr_utils",
        "opensr_utils.data_utils",
        "opensr_utils.model_utils",
        "opensr_utils.cli",
        "opensr_utils.data_utils.transform_to_cog",
    ],
)
def test_imports_are_lightweight_and_have_no_filesystem_side_effects(module, tmp_path):
    source_root = str(Path(__file__).resolve().parents[1])
    environment = {
        **os.environ,
        "PYTHONPATH": source_root,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    script = """
import importlib, pathlib, sys
before = pathlib.Path.cwd()
importlib.import_module(sys.argv[1])
heavy = {'torch', 'numpy', 'rasterio', 'matplotlib', 'opensr_model', 'rio_cogeo', 'requests'}
assert not heavy.intersection(sys.modules), heavy.intersection(sys.modules)
assert pathlib.Path.cwd() == before
assert not list(before.iterdir())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, module],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "module", ["opensr_utils", "opensr_utils.data_utils", "opensr_utils.model_utils"]
)
def test_unknown_public_attribute_raises_attribute_error(module):
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(importlib.import_module(module), "nonexistent_public_api")


def test_top_level_version_is_available():
    import opensr_utils

    assert isinstance(opensr_utils.__version__, str)
    assert opensr_utils.__version__

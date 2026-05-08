import importlib


def test_top_level_import_is_lightweight():
    mod = importlib.import_module("opensr_utils")
    assert hasattr(mod, "__version__")


def test_cog_helper_has_no_import_time_side_effects():
    mod = importlib.import_module("opensr_utils.data_utils.transform_to_cog")
    assert hasattr(mod, "process_tree")

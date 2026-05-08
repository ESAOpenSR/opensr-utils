from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("pytorch_lightning")

from rasterio.windows import Window  # noqa: E402

from opensr_utils.model_utils import prepare_model  # noqa: E402


def parent(tmp_path, dataset_len=2):
    return SimpleNamespace(
        log_dir=str(tmp_path),
        datamodule=SimpleNamespace(dataset=[None] * dataset_len),
        _log=lambda message: None,
    )


def test_srmodelpl_upsamples_with_predict_alias():
    model = prepare_model.SRModelPL(factor=2)
    x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)

    out = model.predict(x)

    assert out.shape == (1, 1, 4, 4)
    assert torch.equal(out, model.forward(x))


def test_srmodelpl_factor_one_is_passthrough():
    model = prepare_model.SRModelPL(factor=1)
    x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)

    assert model.predict(x) is x


def test_get_world_info_prefers_initialized_distributed(monkeypatch):
    monkeypatch.setattr(prepare_model.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(prepare_model.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(prepare_model.torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(prepare_model.torch.distributed, "get_rank", lambda: 2)

    assert prepare_model.get_world_info() == (4, 2)


def test_get_world_info_uses_cuda_count_without_distributed(monkeypatch):
    monkeypatch.setattr(prepare_model.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(
        prepare_model.torch.distributed, "is_initialized", lambda: False
    )
    monkeypatch.setattr(prepare_model.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prepare_model.torch.cuda, "device_count", lambda: 3)

    assert prepare_model.get_world_info() == (3, 0)


def test_preprocess_model_uses_placeholder_and_writes_prediction_index(tmp_path):
    windows = [Window(0, 0, 2, 2), Window(2, 0, 2, 2)]
    model = prepare_model.preprocess_model(
        parent(tmp_path), None, str(tmp_path), windows, factor=2
    )
    model._save_compressed = False
    model._save_progress_preview = False

    model.on_predict_start()
    batch = {
        "image": torch.full((2, 1, 2, 2), 0.5, dtype=torch.float32),
        "meta": {
            "row_off": torch.tensor([0, 0]),
            "col_off": torch.tensor([0, 2]),
            "height": torch.tensor([2, 2]),
            "width": torch.tensor([2, 2]),
            "index": torch.tensor([0, 1]),
        },
    }
    assert model.predict_step(batch, 0) is None
    model.on_predict_end()

    index = (tmp_path / "index.json").read_text()
    assert '"saved_container":"npy"' in index
    assert '"output_bands":1' in index
    assert len(model._entries) == 2
    for entry in model._entries:
        assert entry["bands"] == 1
        arr = np.load(entry["path"])
        assert arr.dtype == np.uint16
        assert arr.shape == (1, 4, 4)
        assert arr.min() == 5000
        assert arr.max() == 5000
    assert not (tmp_path / "index_rank0.json").exists()


def test_preprocess_model_can_save_compressed_npz(tmp_path):
    windows = [Window(1, 2, 2, 2)]
    model = prepare_model.preprocess_model(
        parent(tmp_path, dataset_len=1), None, str(tmp_path), windows, factor=2
    )
    model._save_compressed = True

    model.on_predict_start()
    model.predict_step(
        {
            "image": torch.ones((1, 1, 2, 2), dtype=torch.float32),
            "meta": [{"row_off": 2, "col_off": 1, "height": 2, "width": 2, "index": 0}],
        },
        0,
    )
    model.on_predict_end()

    assert model._entries[0]["path"].endswith(".npz")
    assert model._entries[0]["bands"] == 1
    with np.load(model._entries[0]["path"]) as z:
        assert z["arr"].shape == (1, 4, 4)
    assert '"saved_container":"npz"' in (tmp_path / "index.json").read_text()


def test_preprocess_model_records_non_four_band_outputs(tmp_path):
    class TwoBandModel(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(
                x[:, :2], scale_factor=3, mode="nearest"
            )

    model = prepare_model.preprocess_model(
        parent(tmp_path, dataset_len=1),
        TwoBandModel(),
        str(tmp_path),
        [Window(0, 0, 2, 3)],
        factor=3,
    )
    model.on_predict_start()
    model.predict_step(
        {
            "image": torch.ones((1, 4, 3, 2), dtype=torch.float32),
            "meta": {
                "row_off": [0],
                "col_off": [0],
                "height": [3],
                "width": [2],
                "index": [0],
            },
        },
        0,
    )
    model.on_predict_end()

    assert model._entries[0]["bands"] == 2
    assert model._entries[0]["height_hr"] == 9
    assert model._entries[0]["width_hr"] == 6
    assert '"output_bands":2' in (tmp_path / "index.json").read_text()


def test_preprocess_model_allows_factor_one_identity_model(tmp_path):
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    model = prepare_model.preprocess_model(
        parent(tmp_path, dataset_len=1),
        Identity(),
        str(tmp_path),
        [Window(3, 2, 2, 2)],
        factor=1,
    )
    model.on_predict_start()
    model.predict_step(
        {
            "image": torch.full((1, 1, 2, 2), 0.25, dtype=torch.float32),
            "meta": [
                {"row_off": 2, "col_off": 3, "height": 2, "width": 2, "index": 0}
            ],
        },
        0,
    )
    model.on_predict_end()

    entry = model._entries[0]
    assert entry["row_off_hr"] == 2
    assert entry["col_off_hr"] == 3
    assert entry["height_hr"] == 2
    assert entry["width_hr"] == 2
    np.testing.assert_array_equal(
        np.load(entry["path"]), np.full((1, 2, 2), 2500, dtype=np.uint16)
    )
    assert '"factor":1' in (tmp_path / "index.json").read_text()


def test_preprocess_model_wraps_torch_module_and_rejects_wrong_output_shape(tmp_path):
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    windows = [Window(0, 0, 2, 2)]
    model = prepare_model.preprocess_model(
        parent(tmp_path, dataset_len=1), Identity(), str(tmp_path), windows, factor=2
    )
    model.on_predict_start()

    with pytest.raises(ValueError, match="does not match expected"):
        model.predict_step(
            {
                "image": torch.ones((1, 1, 2, 2), dtype=torch.float32),
                "meta": {
                    "row_off": [0],
                    "col_off": [0],
                    "height": [2],
                    "width": [2],
                    "index": [0],
                },
            },
            0,
        )


def test_preprocess_model_preserves_lightning_module_instance(tmp_path):
    model = prepare_model.SRModelPL(factor=2)
    model.train()

    prepared = prepare_model.preprocess_model(
        parent(tmp_path, dataset_len=1), model, str(tmp_path), [], factor=2
    )

    assert prepared is model
    assert prepared.training is False
    assert hasattr(prepared, "predict_step")

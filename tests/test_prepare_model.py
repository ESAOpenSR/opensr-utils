from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import numpy as np
import torch

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
            "meta": [{"row_off": 2, "col_off": 3, "height": 2, "width": 2, "index": 0}],
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


def prediction_batch(images, *, row=2, col=3, metadata="lists"):
    batch_size, _, height, width = images.shape
    records = [
        dict(row_off=row, col_off=col + i * width, height=height, width=width, index=i)
        for i in range(batch_size)
    ]
    if metadata == "records":
        meta = records
    else:
        meta = {key: [record[key] for record in records] for key in records[0]}
        if metadata == "tensors":
            meta = {key: torch.tensor(values) for key, values in meta.items()}
    return {"image": images, "meta": meta}


@pytest.mark.parametrize("metadata", ["lists", "records", "tensors"])
@pytest.mark.parametrize("compressed", [False, True])
def test_prediction_manifest_round_trips_values_coordinates_and_format(
    tmp_path, metadata, compressed
):
    model = prepare_model.preprocess_model(
        parent(tmp_path), None, str(tmp_path), [], factor=2
    )
    model._save_compressed = compressed
    model.on_predict_start()
    images = torch.stack([torch.full((2, 2, 3), 0.25), torch.full((2, 2, 3), 0.75)])
    assert model.predict_step(prediction_batch(images, metadata=metadata), 0) is None
    model.on_predict_end()
    manifest = json.loads((tmp_path / "index.json").read_text())
    assert manifest["factor"] == 2
    assert manifest["saved_dtype"] == "uint16"
    assert manifest["saved_scale"] == 10000
    assert manifest["saved_container"] == ("npz" if compressed else "npy")
    assert manifest["output_bands"] == 2
    assert len(manifest["entries"]) == 2
    for i, entry in enumerate(manifest["entries"]):
        assert (entry["row_off_lr"], entry["col_off_lr"]) == (2, 3 + i * 3)
        assert (entry["row_off_hr"], entry["col_off_hr"]) == (4, 6 + i * 6)
        assert (entry["height_lr"], entry["width_lr"]) == (2, 3)
        assert (entry["height_hr"], entry["width_hr"], entry["bands"]) == (4, 6, 2)
        if compressed:
            with np.load(entry["path"]) as archive:
                data = archive[manifest["saved_key"]]
        else:
            data = np.load(entry["path"])
        np.testing.assert_array_equal(
            data, np.full((2, 4, 6), [2500, 7500][i], dtype=np.uint16)
        )
    assert not list(tmp_path.glob("*.png")), "Previews are opt-in"


@pytest.mark.parametrize(
    "values,expected",
    [
        ([-0.25, 0, 0.125, 1.25], [0, 0, 1250, 10000]),
        ([100, 1000, 5000, 12000], [100, 1000, 5000, 10000]),
    ],
)
def test_saved_predictions_support_normalized_and_reflectance_outputs(
    tmp_path, values, expected
):
    model = prepare_model.preprocess_model(
        parent(tmp_path, 1), None, str(tmp_path), [], factor=1
    )
    model.on_predict_start()
    model.predict_step(
        prediction_batch(torch.tensor(values).reshape(1, 1, 2, 2).float()), 0
    )
    model.on_predict_end()
    manifest = json.loads((tmp_path / "index.json").read_text())
    patch = np.load(manifest["entries"][0]["path"])
    assert patch.dtype == np.uint16
    np.testing.assert_array_equal(patch.ravel(), expected)


def test_model_band_count_must_remain_constant_across_batches(tmp_path):
    model = prepare_model.preprocess_model(
        parent(tmp_path), None, str(tmp_path), [], factor=1
    )
    model.on_predict_start()
    model.predict_step(prediction_batch(torch.ones((1, 2, 2, 2))), 0)
    with pytest.raises(ValueError, match="band count changed"):
        model.predict_step(prediction_batch(torch.ones((1, 3, 2, 2)), row=4), 1)
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_wrapped_torch_model_is_in_eval_mode_and_weights_are_preserved(tmp_path):
    original = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 1, bias=False), torch.nn.Dropout(0.9)
    )
    original.train()
    weights = original[0].weight.detach().clone()
    model = prepare_model.preprocess_model(
        parent(tmp_path), original, str(tmp_path), [], factor=1
    )
    image = torch.ones((1, 1, 2, 2))
    assert not original.training
    assert not model.training
    torch.testing.assert_close(original[0].weight, weights)
    torch.testing.assert_close(model.predict(image), original(image))
    torch.testing.assert_close(model.predict(image), model.predict(image))


@pytest.mark.parametrize("factor", [0, -1, 1.5, "2", None, True, False])
def test_model_preparation_rejects_invalid_scale_before_writing(tmp_path, factor):
    with pytest.raises(ValueError, match="positive integer"):
        prepare_model.preprocess_model(
            parent(tmp_path), None, str(tmp_path), [], factor=factor
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.known_bug
@pytest.mark.xfail(
    reason="Non-finite predictions are cast to uint16 rather than rejected",
    raises=pytest.fail.Exception,
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_model_output_is_rejected_before_saving(tmp_path, value):
    model = prepare_model.preprocess_model(
        parent(tmp_path), None, str(tmp_path), [], factor=1
    )
    model.on_predict_start()
    with pytest.raises(ValueError, match="(?i)finite|nan|inf"):
        model.predict_step(prediction_batch(torch.full((1, 1, 2, 2), value)), 0)
    assert not list(tmp_path.glob("*.npy"))


@pytest.mark.known_bug
@pytest.mark.xfail(
    reason="A model can silently drop batch samples without validation",
    raises=pytest.fail.Exception,
)
def test_output_batch_size_must_match_input_batch_size(tmp_path):
    class DropsSample(torch.nn.Module):
        def forward(self, x):
            return x[:1]

    model = prepare_model.preprocess_model(
        parent(tmp_path), DropsSample(), str(tmp_path), [], factor=1
    )
    model.on_predict_start()
    with pytest.raises(ValueError, match="(?i)batch"):
        model.predict_step(prediction_batch(torch.ones((2, 1, 2, 2))), 0)


def distributed_context(monkeypatch):
    """Simulate rank metadata; these tests do not claim GPU/process coverage."""
    dist = prepare_model.torch.distributed
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "barrier", lambda: None)
    monkeypatch.setattr(prepare_model.time, "sleep", lambda _: None)


def test_distributed_manifest_merge_sorts_deduplicates_and_accepts_empty_ranks(
    tmp_path, monkeypatch
):
    distributed_context(monkeypatch)
    model = prepare_model.preprocess_model(
        parent(tmp_path), None, str(tmp_path), [], factor=1
    )
    model.on_predict_start()
    model.predict_step(prediction_batch(torch.ones((1, 1, 2, 2)), row=4), 0)
    model.predict_step(prediction_batch(torch.ones((1, 1, 2, 2)), row=0), 1)
    # Duplicate an actual entry in another rank's persisted manifest.
    (tmp_path / "index_rank1.json").write_text(
        json.dumps({"output_bands": 1, "entries": model._entries[:1]})
    )
    model.on_predict_end()
    index = json.loads((tmp_path / "index.json").read_text())
    assert [(e["row_off_lr"], e["col_off_lr"]) for e in index["entries"]] == [
        (0, 3),
        (4, 3),
    ]
    assert not list(tmp_path.glob("index_rank*.json"))


@pytest.mark.parametrize("other_bands", [0, 2, None])
def test_distributed_manifest_merge_handles_empty_mismatched_and_missing_shards(
    tmp_path, monkeypatch, other_bands
):
    distributed_context(monkeypatch)
    model = prepare_model.preprocess_model(
        parent(tmp_path), None, str(tmp_path), [], factor=1
    )
    model.on_predict_start()
    model.predict_step(prediction_batch(torch.ones((1, 1, 2, 2))), 0)
    if other_bands is not None:
        (tmp_path / "index_rank1.json").write_text(
            json.dumps({"output_bands": other_bands, "entries": []})
        )
    if other_bands == 0:
        model.on_predict_end()
        index = json.loads((tmp_path / "index.json").read_text())
        assert index["output_bands"] == 1
        assert len(index["entries"]) == 1
    else:
        exception, message = (
            (RuntimeError, "Missing prediction shard")
            if other_bands is None
            else (ValueError, "disagree on output band count")
        )
        with pytest.raises(exception, match=message):
            model.on_predict_end()
        assert not (tmp_path / "index.json").exists()

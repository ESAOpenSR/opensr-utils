from __future__ import annotations

import pickle

import pytest

import numpy as np
import torch

from rasterio.windows import Window  # noqa: E402

from opensr_utils.data_utils import datamodule  # noqa: E402


def test_sharded_inference_sampler_splits_contiguous_ranges():
    shards = [
        list(datamodule.ShardedInferenceSampler(10, rank=rank, world_size=3))
        for rank in range(3)
    ]

    assert shards == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert [
        len(datamodule.ShardedInferenceSampler(10, rank=r, world_size=3))
        for r in range(3)
    ] == [4, 3, 3]


def test_sharded_inference_sampler_rejects_invalid_rank():
    with pytest.raises(ValueError, match="Bad rank/world_size"):
        datamodule.ShardedInferenceSampler(10, rank=3, world_size=3)


def test_infer_rank_world_uses_environment_when_distributed_not_initialized(
    monkeypatch,
):
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "5")
    monkeypatch.setattr(datamodule.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(datamodule.torch.distributed, "is_initialized", lambda: False)

    assert datamodule._infer_rank_world() == (2, 5)


def test_prediction_dataset_reads_file_windows_and_normalizes(write_raster, tmp_path):
    data = np.array(
        [
            [[0, 5000, 10000, 20000], [1000, 2000, 3000, 4000]],
            [[10000, 0, 5000, 20000], [9000, 8000, 7000, 6000]],
        ],
        dtype=np.uint16,
    )
    tif = write_raster(tmp_path / "input.tif", data=data)
    dataset = datamodule.PredictionDataset(
        input_type="file",
        root=str(tif),
        windows=[Window(1, 0, 2, 2)],
        lr_file_dict={"file": str(tif)},
    )

    sample = dataset[0]

    assert sample["meta"] == {
        "row_off": 0,
        "col_off": 1,
        "height": 2,
        "width": 2,
        "index": 0,
    }
    assert sample["image"].shape == (2, 2, 2)
    assert sample["image"].dtype == torch.float32
    assert torch.all(sample["image"] >= 0)
    assert torch.all(sample["image"] <= 1)
    assert sample["image"][0, 0, 0].item() == pytest.approx(0.5)
    dataset.close()
    assert dataset._src_cache == {}


def test_prediction_dataset_reads_band_dictionary_in_rgb_nir_order(
    write_raster, tmp_path
):
    band_paths = {}
    for band, value in {"B": 2000, "G": 3000, "R": 4000, "NIR": 8000}.items():
        band_paths[band] = str(
            write_raster(tmp_path / f"{band}.tif", data=[[[value] * 3] * 3])
        )
    dataset = datamodule.PredictionDataset(
        input_type="S2GM",
        root=str(tmp_path),
        windows=[Window(0, 0, 2, 2)],
        lr_file_dict=band_paths,
    )

    image = dataset[0]["image"]

    assert image.shape == (4, 2, 2)
    assert image[:, 0, 0].tolist() == pytest.approx([0.4, 0.3, 0.2, 0.8])
    dataset.close()


def test_prediction_dataset_reads_additional_dictionary_bands(write_raster, tmp_path):
    band_paths = {}
    for band, value in {
        "B": 2000,
        "G": 3000,
        "R": 4000,
        "NIR": 8000,
        "SWIR": 11000,
    }.items():
        band_paths[band] = str(
            write_raster(tmp_path / f"{band}.tif", data=[[[value] * 3] * 3])
        )
    dataset = datamodule.PredictionDataset(
        input_type="S2GM",
        root=str(tmp_path),
        windows=[Window(0, 0, 2, 2)],
        lr_file_dict=band_paths,
    )

    image = dataset[0]["image"]

    assert image.shape == (5, 2, 2)
    assert image[:, 0, 0].tolist() == pytest.approx([0.4, 0.3, 0.2, 0.8, 1.0])
    dataset.close()


def test_prediction_dataset_rejects_unknown_input_type(tmp_path):
    dataset = datamodule.PredictionDataset(
        input_type="weird",
        root=str(tmp_path),
        windows=[Window(0, 0, 1, 1)],
        lr_file_dict={},
    )

    with pytest.raises(NotImplementedError, match="not supported"):
        dataset[0]


def test_prediction_dataset_pickle_drops_open_dataset_cache(write_raster, tmp_path):
    tif = write_raster(tmp_path / "input.tif")
    dataset = datamodule.PredictionDataset(
        input_type="file",
        root=str(tif),
        windows=[Window(0, 0, 1, 1)],
        lr_file_dict={"file": str(tif)},
    )
    dataset.get_from_file(0)

    restored = pickle.loads(pickle.dumps(dataset))

    assert restored._src_cache == {}
    dataset.close()


def test_prediction_datamodule_builds_plain_predict_loader(write_raster, tmp_path):
    tif = write_raster(tmp_path / "input.tif")
    dm = datamodule.PredictionDataModule(
        input_type="file",
        root=str(tif),
        windows=[Window(0, 0, 2, 2), Window(2, 0, 2, 2)],
        lr_file_dict={"file": str(tif)},
        batch_size=2,
        num_workers=0,
    )
    dm.setup()

    loader = dm.predict_dataloader()
    batch = next(iter(loader))

    assert len(dm.dataset) == 2
    assert batch["image"].shape == (2, 1, 2, 2)
    assert loader.prefetch_factor is None


def test_prediction_datamodule_uses_sharded_sampler_under_ddp(
    monkeypatch, write_raster, tmp_path
):
    tif = write_raster(tmp_path / "input.tif")
    dm = datamodule.PredictionDataModule(
        input_type="file",
        root=str(tif),
        windows=[Window(i, 0, 1, 1) for i in range(5)],
        lr_file_dict={"file": str(tif)},
        batch_size=1,
        num_workers=0,
    )
    dm.setup()
    monkeypatch.setattr(datamodule.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(datamodule.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(datamodule, "_infer_rank_world", lambda: (1, 2))

    loader = dm.predict_dataloader()

    assert isinstance(loader.sampler, datamodule.ShardedInferenceSampler)
    assert list(loader.sampler) == [3, 4]


@pytest.mark.parametrize("size", [0, 1, 2, 7, 31])
@pytest.mark.parametrize("world", [1, 2, 3, 8])
def test_sampler_covers_each_index_exactly_once_and_balances_shards(size, world):
    samplers = [
        datamodule.ShardedInferenceSampler(size, rank=r, world_size=world)
        for r in range(world)
    ]
    shards = [list(sampler) for sampler in samplers]
    assert [i for shard in shards for i in shard] == list(range(size))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    assert [len(s) for s in samplers] == [len(s) for s in shards]
    assert all(list(sampler) == shard for sampler, shard in zip(samplers, shards))


@pytest.mark.parametrize("rank,world", [(-1, 2), (0, 0), (0, -1), (2, 2)])
def test_sampler_rejects_impossible_rank_world_pairs(rank, world):
    with pytest.raises(ValueError):
        datamodule.ShardedInferenceSampler(5, rank=rank, world_size=world)


@pytest.mark.parametrize("input_type", ["file", "SAFE", "S2GM"])
def test_dataset_clips_reflectance_and_preserves_spatial_values(
    input_type, dataset_factory, write_raster, tmp_path
):
    values = np.array([[-200, 0, 1000], [5000, 10000, 15000]], dtype=np.int16)
    path = write_raster(tmp_path / "signed.tif", data=values, dtype="int16")
    dataset = dataset_factory(
        input_type=input_type,
        root=str(path),
        windows=[Window(0, 0, 3, 2)],
        lr_file_dict={"R": str(path)},
    )
    image = dataset[0]["image"]
    np.testing.assert_allclose(image.numpy(), [[[0, 0, 0.1], [0.5, 1, 1]]])
    assert image.dtype == torch.float32
    with datamodule.rasterio.open(path) as src:
        np.testing.assert_array_equal(src.read(1), values)


@pytest.mark.parametrize("count", [1, 2, 4, 7, 13])
def test_file_dataset_preserves_every_band_in_native_order(
    count, dataset_factory, write_raster, tmp_path
):
    data = np.arange(count * 3 * 5, dtype=np.uint16).reshape(count, 3, 5) * 10
    path = write_raster(tmp_path / "multiband.tif", data=data)
    dataset = dataset_factory(
        input_type="file",
        root=str(path),
        lr_file_dict={},
        windows=[Window(2, 1, 3, 2)],
    )
    np.testing.assert_allclose(
        dataset[0]["image"], data[:, 1:3, 2:5] / 10000, atol=1e-7
    )


@pytest.mark.parametrize("input_type", ["SAFE", "S2GM"])
def test_dictionary_dataset_rejects_empty_bands(input_type, dataset_factory, tmp_path):
    dataset = dataset_factory(
        input_type=input_type,
        root=str(tmp_path),
        windows=[Window(0, 0, 1, 1)],
        lr_file_dict={},
    )
    with pytest.raises(ValueError, match="No input bands"):
        dataset[0]


def test_dataset_can_be_read_after_pickling_and_closing_original(
    dataset_factory, write_raster, tmp_path
):
    path = write_raster(tmp_path / "input.tif")
    original = dataset_factory(
        input_type="file", root=str(path), windows=[Window(1, 2, 2, 3)], lr_file_dict={}
    )
    before = original[0]
    restored = pickle.loads(pickle.dumps(original))
    try:
        original.close()
        original.close()  # Closing twice must be safe.
        assert restored[0]["meta"] == before["meta"]
        torch.testing.assert_close(restored[0]["image"], before["image"])
        torch.testing.assert_close(original[0]["image"], before["image"])
    finally:
        restored.close()


@pytest.mark.parametrize("size,batch_size", [(0, 2), (1, 4), (5, 2), (5, 3)])
def test_prediction_loader_never_drops_or_reorders_windows(
    size, batch_size, write_raster, tmp_path
):
    path = write_raster(tmp_path / "input.tif", width=8, height=2)
    dm = datamodule.PredictionDataModule(
        input_type="file",
        root=str(path),
        windows=[Window(i, 0, 1, 1) for i in range(size)],
        lr_file_dict={},
        batch_size=batch_size,
        num_workers=0,
        prefetch_factor=2,
    )
    dm.setup()
    try:
        batches = list(dm.predict_dataloader())
        assert [int(i) for batch in batches for i in batch["meta"]["index"]] == list(
            range(size)
        )
        assert sum(batch["image"].shape[0] for batch in batches) == size
    finally:
        dm.dataset.close()


def test_initialized_distributed_rank_takes_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "99")
    monkeypatch.setattr(datamodule.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(datamodule.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(datamodule.torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(datamodule.torch.distributed, "get_world_size", lambda: 4)
    assert datamodule._infer_rank_world() == (3, 4)

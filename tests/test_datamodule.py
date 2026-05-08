from __future__ import annotations

import pickle

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("pytorch_lightning")

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

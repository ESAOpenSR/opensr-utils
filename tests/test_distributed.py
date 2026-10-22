"""Small real CPU/Gloo checks for prediction finalization."""

from types import SimpleNamespace
import os
from datetime import timedelta
import queue as queue_module
import time

import pytest
import torch
import torch.multiprocessing as mp

from opensr_utils.model_utils import prepare_model as prepare_module
from opensr_utils.model_utils.prepare_model import get_world_info, preprocess_model


def _worker(rank, world, init_file, output_dir, mode, queue, backend="gloo"):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    if backend == "gloo" and os.name == "posix":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    if backend == "nccl":
        torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=10),
    )
    try:
        parent = SimpleNamespace(
            log_dir=output_dir,
            datamodule=SimpleNamespace(dataset=[None] * world),
            _log=lambda _message: None,
        )
        model = preprocess_model(parent, None, output_dir, [], factor=1)
        model._save_sync_timeout = 3
        model.on_predict_start()
        if mode in ("rankcheck", "missing_shard", "missing_shard_recover"):
            observed_world, observed_rank = get_world_info()
            assert (observed_world, observed_rank) == (world, rank)
        if mode != "empty" or rank == 0:
            batch = {
                "image": torch.ones(
                    (1, 1, 2, 2), device=f"cuda:{rank}" if backend == "nccl" else "cpu"
                ),
                "meta": [{"row_off": rank, "col_off": 0, "height": 2, "width": 2}],
            }
            model.predict_step(batch, 0)
        if mode in ("band_error", "recover") and rank == 1:
            model._output_bands = 2
        if mode == "missing" and rank == 1:
            queue.put((rank, "lost-peer"))
            return
        try:
            original_replace = os.replace
            replace_state = {"failed": False}

            def replace_with_failure(source, destination):
                if (
                    mode in ("write_failure", "write_failure_recover")
                    and rank == 1
                    and not replace_state["failed"]
                    and destination.endswith("index_rank1.json")
                ):
                    replace_state["failed"] = True
                    raise OSError("injected shard write failure")
                return original_replace(source, destination)

            if mode in ("write_failure", "write_failure_recover"):
                os.replace = replace_with_failure
            original_barrier = prepare_module._distributed_barrier
            state = {"first": True}

            def barrier_with_missing_shard(timeout_seconds=30):
                result = original_barrier(timeout_seconds)
                if (
                    rank == 0
                    and mode in ("missing_shard", "missing_shard_recover")
                    and state["first"]
                ):
                    shard = os.path.join(output_dir, "index_rank1.json")
                    if os.path.exists(shard):
                        os.unlink(shard)
                    state["first"] = False
                return result

            if mode in ("missing_shard", "missing_shard_recover"):
                prepare_module._distributed_barrier = barrier_with_missing_shard
            if mode == "recover":
                try:
                    model.on_predict_end()
                except Exception:
                    pass
                model._output_bands = 1
                model.on_predict_end()
            elif mode == "missing_shard_recover":
                try:
                    model.on_predict_end()
                except Exception:
                    pass
                model.on_predict_end()
            elif mode == "write_failure_recover":
                try:
                    model.on_predict_end()
                except Exception:
                    pass
                os.replace = original_replace
                model.on_predict_end()
            else:
                model.on_predict_end()
            queue.put((rank, "ok"))
        except Exception as exc:
            queue.put((rank, str(exc)))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run(tmp_path, mode, backend="gloo"):
    init_file = str(tmp_path / "gloo-init")
    queue = mp.get_context("spawn").Queue()
    context = mp.spawn(
        _worker,
        args=(2, init_file, str(tmp_path), mode, queue, backend),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 45
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Distributed workers exceeded the 45-second deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    results = []
    for _ in range(2):
        try:
            results.append(queue.get(timeout=2))
        except (queue_module.Empty, TypeError):
            results.append((-1, "worker produced no result"))
    queue.close()
    queue.join_thread()
    return results


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed unavailable"
)
def test_real_gloo_handles_empty_rank_and_global_rank_metadata(tmp_path):
    results = _run(tmp_path, "empty")
    assert sorted(results) == [(0, "ok"), (1, "ok")]


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed unavailable"
)
def test_real_gloo_propagates_manifest_merge_failure_to_all_ranks(tmp_path):
    results = _run(tmp_path, "band_error")
    assert all(
        status.startswith("Distributed prediction finalization failed:")
        for _, status in results
    )
    assert all("output band count" in status for _, status in results)


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed unavailable"
)
def test_real_gloo_missing_peer_is_bounded(tmp_path):
    results = _run(tmp_path, "missing")
    assert any(status == "lost-peer" for _, status in results)
    assert any(
        "barrier" in status.lower() or "distributed prediction" in status.lower()
        for _, status in results
    )


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed unavailable"
)
def test_real_gloo_merge_failure_can_recover_in_same_group(tmp_path):
    results = _run(tmp_path, "recover")
    assert sorted(results) == [(0, "ok"), (1, "ok")]


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed unavailable"
)
def test_real_gloo_missing_shard_is_reported_to_all_ranks(tmp_path):
    results = _run(tmp_path, "missing_shard")
    assert all(
        "Distributed prediction finalization failed" in status for _, status in results
    )
    assert all("Missing prediction shard" in status for _, status in results)


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed unavailable"
)
def test_real_gloo_missing_shard_recovers_in_same_group(tmp_path):
    results = _run(tmp_path, "missing_shard_recover")
    assert sorted(results) == [(0, "ok"), (1, "ok")]


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed unavailable"
)
def test_real_gloo_rank1_shard_write_failure_propagates(tmp_path):
    results = _run(tmp_path, "write_failure")
    assert all("injected shard write failure" in status for _, status in results)


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed unavailable"
)
def test_real_gloo_rank1_shard_write_failure_recovers(tmp_path):
    results = _run(tmp_path, "write_failure_recover")
    assert sorted(results) == [(0, "ok"), (1, "ok")]


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="PyTorch distributed unavailable"
)
def test_real_gloo_uses_global_rank_when_local_rank_differs(tmp_path):
    results = _run(tmp_path, "rankcheck")
    assert sorted(results) == [(0, "ok"), (1, "ok")]


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available(),
    reason="requires two CUDA devices and NCCL",
)
def test_real_nccl_prediction_and_empty_rank_when_two_gpus_are_available(tmp_path):
    results = _run(tmp_path, "empty", backend="nccl")
    assert sorted(results) == [(0, "ok"), (1, "ok")]
    import json
    import numpy as np

    manifest = json.loads((tmp_path / "index.json").read_text())
    assert len(manifest["entries"]) == 1
    np.testing.assert_array_equal(
        np.load(manifest["entries"][0]["path"]), np.full((1, 2, 2), 10000)
    )

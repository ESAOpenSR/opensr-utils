"""A real Lightning CPU/DDP prediction check with an empty rank."""

import json
import os
import subprocess
import sys
import socket

import numpy as np
import pytest


@pytest.mark.skipif(os.name == "nt", reason="ddp_fork is Unix-only")
def test_lightning_ddp_fork_persists_one_window_with_empty_rank(tmp_path):
    """Two CPU ranks must finish when the global sampler has one item."""
    raster = tmp_path / "input.tif"
    temp = tmp_path / "temp"
    script = r"""
import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from opensr_utils.data_utils.datamodule import PredictionDataModule
from opensr_utils.model_utils.prepare_model import preprocess_model

raster_path, temp_path = sys.argv[1:]
with rasterio.open(
    raster_path, "w", driver="GTiff", width=2, height=2, count=1,
    dtype="uint16", transform=from_origin(0, 2, 1, 1), nodata=None,
) as dst:
    dst.write(np.array([[[1000, 2000], [3000, 4000]]], dtype=np.uint16))

windows = [Window(0, 0, 2, 2)]
dm = PredictionDataModule(
    "file", raster_path, windows, {"file": raster_path},
    batch_size=1, num_workers=0,
)
dm.setup()
parent = SimpleNamespace(log_dir=temp_path, datamodule=dm, _log=lambda message: None)
model = preprocess_model(parent, None, temp_path, windows, factor=1)
model.model_dm_length = len(dm.dataset)
model._save_temp_folder = temp_path
model._save_windows = windows
model._save_factor = 1
model._save_compressed = False
model._save_progress_preview = False
model._save_sync_timeout = 20

trainer = Trainer(
    accelerator="cpu", devices=2,
    strategy=DDPStrategy(start_method="fork", timeout=timedelta(seconds=10)),
    logger=False,
    enable_checkpointing=False, enable_model_summary=False,
    enable_progress_bar=False, inference_mode=True,
    use_distributed_sampler=False,
)
trainer.predict(model, datamodule=dm, return_predictions=False)
"""
    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    # Keep concurrent pytest workers from sharing a process-group endpoint.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        env["MASTER_PORT"] = str(listener.getsockname()[1])
    if sys.platform.startswith("linux"):
        env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    env.setdefault("MPLCONFIGDIR", str(tmp_path / "mpl"))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(raster), str(temp)],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    index_path = temp / "index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text())
    assert len(index["entries"]) == 1
    patch_path = temp / "sr_r000000000_c000000000.npy"
    assert patch_path.exists()
    patch = np.load(patch_path)
    assert patch.shape == (1, 2, 2)
    np.testing.assert_array_equal(patch[0], [[1000, 2000], [3000, 4000]])

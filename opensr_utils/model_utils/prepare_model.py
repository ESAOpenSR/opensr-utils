import os
import json
import torch
import numpy as np
from pytorch_lightning import LightningModule

# -------------------------------------------------------------------------
# Placeholder interpolation model
class SRModelPL(LightningModule): 
    def __init__(self):
        super(SRModelPL, self).__init__()
    def forward(self, x):
        sr = torch.nn.functional.interpolate(x, size=(512, 512), mode='nearest')
        return sr
    def predict(self,x):
        return self.forward(x)

# -------------------------------------------------------------------------
# Preprocessor that also attaches predict_step + hooks that save predictions to disk
def preprocess_model(model,temp_folder,windows,factor):
    """
    Returns a LightningModule with a predict_step that:
      - runs inference
      - saves each patch to .npy in model._save_temp_folder
      - writes a per-rank index json when prediction ends

    Before calling Trainer.predict(), you must set on the model:
      model._save_temp_folder = "temp/"
      model._save_windows     = list of LR rasterio.windows.Window
      model._save_factor      = int (SR scale)
    """
    
    
    # --- define hooks here so they stay out of the global namespace ---
    def _rank_world(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def on_predict_start(self):
        assert hasattr(self, "_save_temp_folder"), "set model._save_temp_folder"
        assert hasattr(self, "_save_windows"), "set model._save_windows"
        assert hasattr(self, "_save_factor"), "set model._save_factor"
        os.makedirs(self._save_temp_folder, exist_ok=True)
        self._rank, self._world = _rank_world(self)
        self._local_idx = 0
        self._entries = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch if torch.is_tensor(batch) else batch["image"]
        sr = self.predict(x) if hasattr(self, "predict") else self.forward(x)
        sr = sr.detach().cpu()
        scale = int(self._save_factor)
        B, C, Hh, Ww = sr.shape

        for i in range(B):
            win = self._save_windows[self._local_idx]
            self._local_idx += 1

            row_off_hr = int(win.row_off) * scale
            col_off_hr = int(win.col_off) * scale
            arr = sr[i].numpy().astype(np.float32, copy=False)

            fn = f"sr_r{row_off_hr:09d}_c{col_off_hr:09d}.npy"
            out_path = os.path.join(self._save_temp_folder, fn)
            np.save(out_path, arr)

            self._entries.append({
                "path": out_path,
                "row_off_lr": int(win.row_off), "col_off_lr": int(win.col_off),
                "height_lr": int(win.height),   "width_lr":  int(win.width),
                "row_off_hr": row_off_hr,       "col_off_hr": col_off_hr,
                "height_hr": int(Hh),           "width_hr":  int(Ww),
            })

        return None  # prevents Lightning from gathering predictions in memory

    def on_predict_end(self):
        shard = {
            "rank": self._rank,
            "world_size": self._world,
            "factor": int(self._save_factor),
            "entries": self._entries,
        }
        shard_path = os.path.join(self._save_temp_folder, f"index_rank{self._rank}.json")
        with open(shard_path, "w") as f:
            json.dump(shard, f)
        if self._rank == 0:
            print(f"[rank {self._rank}] wrote index shard: {shard_path}")

    # ---------------------------------------------------------------------
    # Attach hooks to model
    def _attach_hooks(m):
        m.on_predict_start = on_predict_start.__get__(m, m.__class__)
        m.predict_step = predict_step.__get__(m, m.__class__)
        m.on_predict_end = on_predict_end.__get__(m, m.__class__)
        return m

    # --- wrap model if needed ---
    if isinstance(model, LightningModule):
        print("Model is a LightningModule.")
        model.eval()
    elif isinstance(model, torch.nn.Module):
        print("Model is a torch.nn.Module. Wrapping in LightningModule.")
        class WrappedModel(LightningModule):
            def __init__(self, mdl):
                super().__init__()
                self.model = mdl
            def forward(self, x): return self.model(x)
            def predict(self, x): return self.forward(x)
        wrapped_model = WrappedModel(model)
        wrapped_model.eval()
        model = wrapped_model
    elif model is None:
        print("No model provided. Using placeholder interpolation model.")
        placeholder_model = SRModelPL()
        placeholder_model.eval()
        model = placeholder_model
    else:
        raise NotImplementedError("Model must be a LightningModule, torch.nn.Module, or None.")


    # set model variables
    model = _attach_hooks(model)
    model._save_temp_folder = temp_folder
    model._save_windows = windows
    model._save_factor = factor
    return model
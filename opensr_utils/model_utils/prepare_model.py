import os
import json
import torch
import numpy as np
from pytorch_lightning import LightningModule
import time
import glob


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
        x    = batch["image"]                        # (B,C,H,W) tensor
        meta = batch["meta"]                         # list[dict] or dict of lists
        sr = self.predict(x) if hasattr(self, "predict") else self.forward(x)
        sr = sr.detach().cpu()                       # (B,C,Hh,Ww)

        scale = int(self._save_factor)
        B, C, Hh, Ww = sr.shape

        # meta may be a dict of lists (PL default); normalize to per-sample dicts
        if isinstance(meta, dict) and isinstance(meta["row_off"], torch.Tensor):
            meta = [{k: int(v[i].item()) for k, v in meta.items()} for i in range(B)]
        elif isinstance(meta, dict):
            meta = [{k: int(v[i]) for k, v in meta.items()} for i in range(B)]

        for i in range(B):
            mi = meta[i]
            row_off_lr = mi["row_off"]; col_off_lr = mi["col_off"]
            row_off_hr = row_off_lr * scale; col_off_hr = col_off_lr * scale

            s = sr[i]                                # (C,Hh,Ww) float
            mean_val = float(torch.mean(s))
            # simple range heuristic: [0,1] vs already ~[0,10000]
            if mean_val < 10:
                s = torch.clamp(s, 0.0, 1.0) * 10000.0
            else:
                s = torch.clamp(s, 0.0, 10000.0)

            s = torch.round(s)
            arr = s.numpy().astype(np.uint16, copy=False)

            fn = f"sr_r{row_off_hr:09d}_c{col_off_hr:09d}.npz"
            out_path = os.path.join(self._save_temp_folder, fn)
            np.savez_compressed(out_path, arr=arr)   # direct compressed save

            self._entries.append({
                "path": out_path,
                "row_off_lr": row_off_lr, "col_off_lr": col_off_lr,
                "height_lr":  mi["height"], "width_lr": mi["width"],
                "row_off_hr": row_off_hr,   "col_off_hr": col_off_hr,
                "height_hr":  int(Hh),      "width_hr":  int(Ww),
            })
        return None

    def on_predict_end(self):
        shard = {
            "rank": self._rank,
            "world_size": self._world,
            "factor": int(self._save_factor),
            "entries": self._entries,
            "saved_dtype": "uint16",
            "saved_scale": 10000,
            "saved_container": "npz",
            "saved_key": "arr",
        }
        shard_path = os.path.join(self._save_temp_folder, f"index_rank{self._rank}.json")
        with open(shard_path, "w") as f:
            json.dump(shard, f, separators=(",", ":"))  # compact
            f.flush()
            os.fsync(f.fileno())
        print(f"[rank {self._rank}] wrote {len(self._entries)} entries → {shard_path}")

        # ---- DDP sync ----
        ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        if ddp:
            torch.distributed.barrier()

        # ---- Merge on rank 0 (or single-process) ----
        if (not ddp) or (torch.distributed.get_rank() == 0):
            world = self._world if ddp else 1
            shard_paths = [os.path.join(self._save_temp_folder, f"index_rank{r}.json") for r in range(world)]

            # (Optional) wait briefly if FS is slow
            for _ in range(50):
                if all(os.path.exists(p) for p in shard_paths): break
                time.sleep(0.05)

            merged_entries = []
            meta = {
                "factor": int(self._save_factor),
                "saved_dtype": "uint16",
                "saved_scale": 10000,
                "saved_container": "npz",
                "saved_key": "arr",
            }
            for p in shard_paths:
                if not os.path.exists(p):
                    continue  # tolerate empty shard or failed worker
                with open(p) as f:
                    data = json.load(f)
                merged_entries.extend(data.get("entries", []))

            # dedupe & deterministic order
            # key by LR offsets + path (handles accidental dupes)
            seen = set()
            uniq = []
            for e in merged_entries:
                k = (e["row_off_lr"], e["col_off_lr"], e["path"])
                if k not in seen:
                    seen.add(k)
                    uniq.append(e)
            uniq.sort(key=lambda e: (e["row_off_lr"], e["col_off_lr"]))

            merged = {**meta, "entries": uniq}
            merged_path = os.path.join(self._save_temp_folder, "index.json")
            with open(merged_path, "w") as f:
                json.dump(merged, f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            print(f"[rank 0] merged {len(uniq)} entries → {merged_path}")
        
        # CleanUp
        try:
            KEEP_SHARDS = False  # toggle for debugging
            KEEP_PARTS = False   # toggle for debugging
            if not KEEP_SHARDS:
                for p in shard_paths:
                    try: os.remove(p)
                    except FileNotFoundError: pass
                    
            # optional: nuke partials
            if KEEP_PARTS==False:
                for p in glob.glob(os.path.join(self._save_temp_folder, "*.part")):
                    try: os.remove(p)
                    except FileNotFoundError: pass
        except UnboundLocalError:
            pass # If we're in single GPU, or in CPU, these variables dont exist


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
#!/usr/bin/env python3
"""
demo.py — Minimal usage example for OpenSR

This script shows:
  1. How to instantiate an SR model from opensr-model
  2. How to run large-scale inference with opensr-utils
"""
import requests
from omegaconf import OmegaConf
from io import StringIO


# --- 1) Create Model -------------------------------------------------
device = "cuda"  # or "cpu" - Dont use the automated detection, it messes up the lightning trainer multi-GPU setup
from opensr_utils.model_utils.get_models import get_ldsrs2
model = get_ldsrs2()

# --- 2) Run large-scale Inference ------------------------------------
import opensr_utils

# Path can be a single .tif, a Sentinel-2 .SAFE folder, or S2GM folder
path = "/path/to/your/input_data"

sr_object = opensr_utils.large_file_processing(
    root=path,                 # File or Folder path
    model=model,               # your SR model
    window_size=(128, 128),    # LR window size for patching
    factor=4,                  # SR factor (10m → 2.5m)
    overlap=12,                # overlapping pixels to avoid artifacts
    eliminate_border_px=2,     # discard border pixels
    device=device,             # "cuda" for GPU-accelerated inference
    gpus=0,                    # pass GPU ID or list of GPUs
)

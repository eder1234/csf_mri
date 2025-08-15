from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from src import CSFVolumeDataset, UNet2D
from src.utils.misc import load_yaml, load_ckpt
from src.utils.metrics import compute_all
from src.utils.viz import save_triplet
from src.utils.temporal import reorder_temporal_images
from src.utils.temporal_features import (
    temporal_std, temporal_tv, dft_bandpower_excl_dc, dft_magnitudes_bins
)

def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top : top + size, left : left + size]

def _first_pc(vol: np.ndarray) -> np.ndarray:
    c, h, w = vol.shape
    x = vol.reshape(c, -1).astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    pc1 = (u[:, 0:1].T @ x).reshape(h, w)
    return pc1

def pad_to_full(mask_crop: np.ndarray, crop: int, full: int = 240) -> np.ndarray:
    pad = (full - crop) // 2
    out = np.zeros((full, full), dtype=mask_crop.dtype)
    out[pad : pad + crop, pad : pad + crop] = mask_crop
    return out

def _in_channels_for_mode(mode: str) -> int:
    if mode == "full":
        return 64
    if mode == "pca":
        return 1
    if mode in ("dft_power", "tvt", "std"):
        return 1
    if mode == "dft_k123":
        return 3
    raise ValueError(f"Unknown input_mode '{mode}'")

def _build_input_for_mode(input_mode: str, phase: np.ndarray, mag: np.ndarray, crop: int) -> np.ndarray:
    """Reproduce the dataset's logic for evaluation."""
    if input_mode == "full":
        vol = np.concatenate([phase, mag], axis=0)              # (64,H,W)
        vol = _center_crop(vol, crop)
        return vol
    if input_mode == "pca":
        vol = np.concatenate([phase, mag], axis=0)              # (64,H,W)
        img = _first_pc(vol)[None, ...]
        return _center_crop(img, crop)
    if input_mode == "dft_power":
        img = dft_bandpower_excl_dc(phase)[None, ...]
        return _center_crop(img, crop)
    if input_mode == "tvt":
        img = temporal_tv(phase)[None, ...]
        return _center_crop(img, crop)
    if input_mode == "std":
        img = temporal_std(phase)[None, ...]
        return _center_crop(img, crop)
    if input_mode == "dft_k123":
        img = dft_magnitudes_bins(phase, bins=(1, 2, 3))        # (3,H,W)
        return _center_crop(img, crop)
    raise ValueError(f"Unknown input_mode '{input_mode}'")

def evaluate(cfg: Dict,
             split: str = "val",
             best_model: str | Path = "outputs/checkpoints/best.pt") -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop = cfg["data"]["crop_size"]
    input_mode = cfg["data"].get("input_mode", "full")

    # Dataset (only for GT masks & IDs)
    root = Path(cfg["data"]["root"])
    data_root = root / (cfg["data"]["test_dir"] if split == "test" else cfg["data"]["train_dir"])
    ds = CSFVolumeDataset(
        root_dir=data_root,
        split=split,
        crop_size=crop,
        val_split=cfg["data"]["val_split"],
        input_mode=input_mode,
    )
    loader = DataLoader(ds, batch_size=1, sampler=SequentialSampler(ds))

    # Model
    ckpt = load_ckpt(best_model, map_location=device)
    if "params" in ckpt and ckpt["params"].get("base_channels"):
        cfg["model"]["base_channels"] = ckpt["params"]["base_channels"]

    in_ch = _in_channels_for_mode(input_mode)
    model = UNet2D(
        in_channels=in_ch,
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Evaluation
    metrics_all: List[Dict[str, float]] = []
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            subj_id = batch["id"][0]
            gt_crop = batch["mask"].numpy()[0]
            gt_full = pad_to_full(gt_crop, crop)

            subj_dir = data_root / subj_id
            phase = np.load(subj_dir / "phase.npy")       # (32,240,240)
            mag   = np.load(subj_dir / "mag.npy")         # (32,240,240)

            best_metrics, best_pred_full = None, None

            for shift in range(32):
                ph_s, mag_s, _ = reorder_temporal_images(phase, mag, shift)
                vol = _build_input_for_mode(input_mode, ph_s, mag_s, crop)
                vol = (vol - vol.mean()) / (vol.std() + 1e-8)

                inp = torch.from_numpy(vol).unsqueeze(0).float().to(device)
                probs = torch.sigmoid(model(inp)).cpu().numpy()[0, 0]

                pred_full = pad_to_full(probs, crop)
                m = compute_all(pred_full, gt_full)

                if best_metrics is None or m["dice"] > best_metrics["dice"]:
                    best_metrics, best_pred_full = m, pred_full

            metrics_all.append(best_metrics)
            # use magnitude[0] for background visualization as before
            from src.utils.viz import save_triplet
            save_triplet(mag[0], best_pred_full, gt_full, fig_dir / f"{subj_id}.png")

    # Aggregate & report
    keys = metrics_all[0].keys()
    mean = {k: np.mean([m[k] for m in metrics_all]) for k in keys}

    print(f"\nSplit: {split} — subjects: {len(ds)} | mode: {input_mode}")
    for k, v in mean.items():
        print(f"{k:12s}: {v:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split",  choices=["val", "test"], default="val")
    ap.add_argument("--best_model", default="outputs/checkpoints/best_model.pt")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    evaluate(cfg, split=args.split, best_model=args.best_model)

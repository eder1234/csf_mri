import os
import random
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from src.utils.temporal import reorder_temporal_images
from src.utils.temporal_features import (
    temporal_std,
    temporal_tv,
    dft_bandpower_excl_dc,
    dft_magnitudes_bins,
)

def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Crop a square patch of `size×size` centred in `arr` (H and W dims)."""
    h, w = arr.shape[-2:]
    top = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top : top + size, left : left + size]

def _first_pc(vol: np.ndarray) -> np.ndarray:
    """
    Compute the first principal-component image of shape (H, W) from
    `vol` with shape (C, H, W).  Centre voxels across the channel axis,
    run SVD on the C×N matrix (N = H×W), then project.
    """
    c, h, w = vol.shape
    x = vol.reshape(c, -1).astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    w_vec = u[:, 0:1]                     # (C,1)
    pc1 = (w_vec.T @ x).reshape(h, w)     # (H,W)
    return pc1

def _feature_mode_to_channels(mode: str) -> int:
    """Return number of channels for each input_mode."""
    if mode == "full":
        return 64
    if mode == "pca":
        return 1
    if mode in ("dft_power", "tvt", "std"):
        return 1
    if mode == "dft_k123":
        return 3
    raise ValueError(f"Unknown input_mode '{mode}'")

class CSFVolumeDataset(Dataset):
    """
    One sample == one *subject* (not one slice).
    Each subject folder is expected to contain
        phase.npy  (32, 240, 240)
        mag.npy    (32, 240, 240)
        mask.npy   (240, 240)
    Output (depends on input_mode):
        - 'full'      : x  (64, crop, crop) = [32 phase + 32 mag]
        - 'pca'       : x  (1,  crop, crop) = PC1 of (64-stack)
        - 'dft_power' : x  (1,  crop, crop) = DFT band power of phase (excl. DC)
        - 'tvt'       : x  (1,  crop, crop) = temporal total variation of phase
        - 'std'       : x  (1,  crop, crop) = temporal std of phase
        - 'dft_k123'  : x  (3,  crop, crop) = |DFT| at k={1,2,3} for phase
        y  : float32 tensor (1,  crop, crop)     single 2-D mask
        id : folder name (string)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        crop_size: int = 64,
        val_split: float = 0.2,
        seed: int = 42,
        input_mode: str = "full",
        augment_cfg: Optional[dict] = None,
    ):
        super().__init__()

        root_dir = Path(root_dir)
        all_subjects: List[Path] = sorted([p for p in root_dir.iterdir() if p.is_dir()])

        # reproducible subject-level split
        rng = random.Random(seed)
        rng.shuffle(all_subjects)
        val_count = math.ceil(len(all_subjects) * val_split)

        if split == "train":
            self.subjects = all_subjects[val_count:]
        elif split == "val":
            self.subjects = all_subjects[:val_count]
        elif split == "test":
            self.subjects = all_subjects
        else:
            raise ValueError(f"Unknown split '{split}'. Use train / val / test.")

        self.input_mode = input_mode
        self.crop_size = crop_size
        self.split = split
        self.augment_cfg = augment_cfg or {}

        # quick sanity on channels
        _ = _feature_mode_to_channels(self.input_mode)

    def __len__(self) -> int:
        return len(self.subjects)

    def _augment(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply identical spatial transforms to `img` and `mask`."""
        cfg = self.augment_cfg
        if not cfg or self.split != "train":
            return img, mask

        # Random flip
        if cfg.get("flip", False):
            if random.random() < 0.5:  # horizontal
                img = torch.flip(img, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if random.random() < 0.5:  # vertical
                img = torch.flip(img, dims=[-2])
                mask = torch.flip(mask, dims=[-2])

        # Random affine (rotation + translation, no scaling)
        rot_deg = cfg.get("rotation", 0)
        trans_frac = cfg.get("translate", 0.0)
        if rot_deg > 0 or trans_frac > 0:
            angle = random.uniform(-rot_deg, rot_deg)
            max_trans = trans_frac * self.crop_size
            translate = (
                random.uniform(-max_trans, max_trans),
                random.uniform(-max_trans, max_trans),
            )
            img = F.affine(img, angle=angle, translate=translate, scale=1.0, shear=0)
            mask = F.affine(mask, angle=angle, translate=translate, scale=1.0, shear=0)

        # Gaussian noise (image only)
        noise_std = cfg.get("gaussian_noise", 0)
        if noise_std > 0:
            img = img + noise_std * torch.randn_like(img)

        return img, mask

    def _build_input(self, phase: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Build input tensor (C,H,W) based on input_mode. Feature modes use PHASE ONLY.
        """
        mode = self.input_mode
        if mode == "full":
            return np.concatenate([phase, mag], axis=0)  # (64,H,W)

        # build a 64-stack for PCA path
        if mode == "pca":
            vol = np.concatenate([phase, mag], axis=0)  # (64,H,W)
            return _first_pc(vol)[None, ...]            # (1,H,W)

        # --- Shift-invariant temporal features on PHASE only (T=32,H,W) ---
        # Ensure time-first convention (already true: (32,H,W))
        if mode == "dft_power":
            feat = dft_bandpower_excl_dc(phase)
            return feat[None, ...]  # (1,H,W)

        if mode == "tvt":
            feat = temporal_tv(phase)
            return feat[None, ...]  # (1,H,W)

        if mode == "std":
            feat = temporal_std(phase)
            return feat[None, ...]  # (1,H,W)

        if mode == "dft_k123":
            feats = dft_magnitudes_bins(phase, bins=(1, 2, 3))  # (3,H,W)
            return feats

        raise ValueError(f"Unknown input_mode '{mode}'")

    def __getitem__(self, idx: int) -> dict:
        subj_dir: Path = self.subjects[idx]

        phase = np.load(subj_dir / "phase.npy")  # (32, 240, 240)
        mag = np.load(subj_dir / "mag.npy")      # (32, 240, 240)
        mask = np.load(subj_dir / "mask.npy")    # (240, 240)

        # optional temporal cyclic shift augmentation (training only)
        if self.split == "train":
            shift = random.randint(0, 31)  # cyclic shift in [0, 31]
            phase, mag, _ = reorder_temporal_images(phase, mag, shift=shift)

        # Build input according to mode
        img = self._build_input(phase, mag)      # (C,H,W) or (64,H,W)

        # centre-crop to (crop_size, crop_size)
        img = _center_crop(img, self.crop_size)
        mask = _center_crop(mask, self.crop_size)

        # to tensors
        img_t = torch.from_numpy(img).float()           # (C, crop, crop)
        mask_t = torch.from_numpy(mask).unsqueeze(0)    # (1, crop, crop)

        img_t, mask_t = self._augment(img_t, mask_t)

        # normalise image (per-sample, zero-mean unit-var)
        img_t = (img_t - img_t.mean()) / (img_t.std() + 1e-8)

        return {
            "image": img_t,            # FloatTensor
            "mask": mask_t.float(),    # FloatTensor (0 / 1)
            "id": subj_dir.name,
        }

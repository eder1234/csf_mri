"""
Hyper‑parameter optimisation for UNet2D spinal‑CSF segmentation.

* One fixed 80 / 20 train‑validation split (same random seed in `CSFVolumeDataset`).
* Search space (8 configurations):
    – learning‑rate   ∈ {3e‑4, 1e‑3, 3e‑3}
    – base_channels   ∈ {16, 32}
    – LR scheduler    ∈ {"constant", "onecycle"}
* Early stopping: patience = 5 epochs, minimum ΔDice >= 0.003 (≈ +0.3 pp Dice).
* Optuna + Hyperband (ASHA) pruner keeps budget ≤ 2 GPU‑hours.

Usage – from project root:
    python -m src.training.tune --config config.yaml --trials 8 --timeout 7200

Outputs:
    ./outputs/tuning/
        trial_#/best.pt   – best checkpoint per trial
        best_overall.pt   – checkpoint from best trial (lowest val Dice loss)
        study.db          – Optuna SQLite study for reproducibility
"""
from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler
import optuna
from optuna.pruners import HyperbandPruner
from optuna.trial import Trial

# local imports ------------------------------------------------------------
from src import CSFVolumeDataset, UNet2D
from src.utils.misc import load_yaml, seed_everything, save_ckpt
from src.utils.losses import DiceLoss, FlowDiceLoss


def _get_loss_fn(cfg: Dict) -> torch.nn.Module:
    name = cfg["train"].get("loss", "dice").lower()
    if name == "dice":
        return DiceLoss()
    if name == "flow_dice":
        lam = cfg["train"].get("flow_lambda", 0.1)
        return FlowDiceLoss(lambda_flow=lam)
    raise ValueError(f"Unknown loss '{name}'")

# -------------------------------------------------------------------------
# Helper: single epoch run (adapted from train.py)
# -------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler | None,
    train: bool = True,
) -> float:
    model.train(train)
    running: float = 0.0
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        extra = {}
        if "phase" in batch:
            extra["phase"] = batch["phase"].to(device, non_blocking=True)
        if "v_enc" in batch:
            extra["v_enc"] = batch["v_enc"].to(device, non_blocking=True)
        if "pixel_size" in batch:
            extra["pixel_size"] = batch["pixel_size"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, masks, **extra)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        running += loss.item()
    return running / len(loader)

# -------------------------------------------------------------------------
# Objective for Optuna
# -------------------------------------------------------------------------

def objective(trial: Trial, cfg: Dict, device: torch.device, out_dir: Path) -> float:
    start_time = time.time()

    # ---------------- search space ---------------------------------------
    lr = trial.suggest_categorical("lr", [3e-4, 1e-3, 3e-3])
    base_ch = trial.suggest_categorical("base_channels", [16, 32])
    scheduler_name = trial.suggest_categorical("scheduler", ["constant", "onecycle"])

    # ---------------- dataset -------------------------------------------
    use_flow = cfg["train"].get("loss") == "flow_dice"
    metadata_csv = cfg["data"].get("metadata_csv")
    train_ds = CSFVolumeDataset(
        root_dir=Path(cfg["data"]["root"]) / cfg["data"]["train_dir"],
        split="train",
        crop_size=cfg["data"]["crop_size"],
        val_split=cfg["data"]["val_split"],
        augment_cfg=cfg["augment"],
        return_phase=use_flow,
        metadata_csv=metadata_csv,
    )
    val_ds = CSFVolumeDataset(
        root_dir=Path(cfg["data"]["root"]) / cfg["data"]["train_dir"],
        split="val",
        crop_size=cfg["data"]["crop_size"],
        val_split=cfg["data"]["val_split"],
        return_phase=use_flow,
        metadata_csv=metadata_csv,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              sampler=RandomSampler(train_ds), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=SequentialSampler(val_ds),
                            num_workers=2, pin_memory=True)

    # ---------------- model & optimiser ----------------------------------
    model = UNet2D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=base_ch,
    ).to(device)
    criterion = _get_loss_fn(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # LR scheduler ---------------------------------------------------------
    if scheduler_name == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=cfg["train"]["epochs"],
            steps_per_epoch=steps_per_epoch, pct_start=0.3)
    else:
        scheduler = None

    scaler = GradScaler() if cfg["train"].get("mixed_precision", True) else None

    # Early stopping params
    patience = 5
    min_delta = 0.003  # Dice loss (≈ 0.3 pp Dice)

    best_val = math.inf
    epochs_no_improve = 0
    total_epochs = cfg["train"]["epochs"]

    for epoch in range(1, total_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device, scaler=None, train=False)

        # report to Optuna & maybe prune
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # early stopping by val performance
        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            # save checkpoint --------------------------------------------
            ckpt_dir = out_dir / f"trial_{trial.number}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_ckpt({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "params": {"lr": lr, "base_channels": base_ch, "scheduler": scheduler_name},
            }, ckpt_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # log runtime
    trial.set_user_attr("runtime_min", (time.time() - start_time) / 60)

    return best_val

# -------------------------------------------------------------------------
# Main – orchestrates study & final model export
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--trials", type=int, default=8, help="Number of Optuna trials (max configurations)")
    parser.add_argument("--timeout", type=int, default=7200, help="Wall‑clock time budget in seconds")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything()

    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path("outputs") / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    pruner = HyperbandPruner(max_resource=cfg["train"]["epochs"], reduction_factor=3, min_resource=1)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name="unet2d_csf",
                                storage=f"sqlite:///{out_dir / 'study.db'}", load_if_exists=True)

    study.optimize(lambda tr: objective(tr, cfg, device, out_dir), n_trials=args.trials, timeout=args.timeout, show_progress_bar=True)

    print("\n===== Optuna summary =====")
    print("Best trial #{}  val_loss={:.4f}".format(study.best_trial.number, study.best_value))
    print("Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k:14s}: {v}")

    # copy best checkpoint to convenient path
    best_trial_dir = out_dir / f"trial_{study.best_trial.number}"
    best_ckpt = best_trial_dir / "best.pt"
    final_ckpt = out_dir / "best_overall.pt"
    if best_ckpt.exists():
        import shutil
        shutil.copy2(best_ckpt, final_ckpt)
        print(f"\nBest checkpoint copied to {final_ckpt}")
    else:
        print("Warning: best checkpoint not found – something went wrong.")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()

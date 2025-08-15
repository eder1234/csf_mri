from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

# local imports
from src import CSFVolumeDataset, UNet2D
from src.utils.misc import load_yaml, seed_everything, save_ckpt
from src.utils.losses import DiceLoss

def _in_channels_for_mode(mode: str) -> int:
    if mode == "full":
        return 64
    if mode == "pca":
        return 1
    if mode in ("dft_power", "tvt", "std"):
        return 1
    if mode == "dft_k123":
        return 3
    # default to 64 to be safe
    raise ValueError(f"Unknown input_mode '{mode}'")

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    train: bool = True,
    log_every: int = 10,
) -> float:
    model.train(train)
    running_loss = 0.0
    for step, batch in enumerate(loader, 1):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, masks)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        if train and step % log_every == 0:
            print(f"Epoch {epoch} | step {step}/{len(loader)} | loss {loss.item():.4f}")

    return running_loss / len(loader)

def get_model_name(cfg: Dict, custom_name: str | None = None) -> str:
    if custom_name:
        return custom_name
    input_mode = cfg["data"].get("input_mode", "full")
    crop_size = cfg["data"]["crop_size"]
    base_channels = cfg["model"]["base_channels"]
    return f"unet2d_{input_mode}_c{crop_size}_b{base_channels}"

def main(cfg: Dict, model_name: str | None = None) -> None:
    seed_everything()
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model_name = get_model_name(cfg, model_name)
    print(f"Training model: {model_name}")

    input_mode = cfg["data"].get("input_mode", "full")

    # datasets & loaders --------------------------------------------------
    train_ds = CSFVolumeDataset(
        root_dir=Path(cfg["data"]["root"]) / cfg["data"]["train_dir"],
        split="train",
        crop_size=cfg["data"]["crop_size"],
        val_split=cfg["data"]["val_split"],
        input_mode=input_mode,
        augment_cfg=cfg["augment"],
    )
    val_ds = CSFVolumeDataset(
        root_dir=Path(cfg["data"]["root"]) / cfg["data"]["train_dir"],
        split="val",
        crop_size=cfg["data"]["crop_size"],
        val_split=cfg["data"]["val_split"],
        input_mode=input_mode
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        sampler=RandomSampler(train_ds),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=SequentialSampler(val_ds),
        num_workers=2,
        pin_memory=True,
    )
    print(f"Train subjects: {len(train_ds)} | Val subjects: {len(val_ds)}")

    # model, loss, opt ----------------------------------------------------
    in_ch = _in_channels_for_mode(input_mode)
    model = UNet2D(
        in_channels=in_ch,
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    # AMP scaler ----------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler() if cfg["train"]["mixed_precision"] else None

    # logging -------------------------------------------------------------
    out_dir = Path("outputs") / model_name
    tb_writer = SummaryWriter(out_dir / "logs")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # early stopping ------------------------------------------------------
    best_val = math.inf
    patience = 20
    epochs_no_improve = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, train=True
        )
        val_loss = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device, scaler=None, epoch=epoch, train=False
        )

        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)

        # save best -------------------------------------------------------
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            from src.utils.misc import save_ckpt
            save_ckpt(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_dir / "best_model.pt",
            )
            print(f"  ✔ New best val loss {val_loss:.4f} — checkpoint saved.")
        else:
            epochs_no_improve += 1

        # early stopping --------------------------------------------------
        if cfg["train"]["early_stopping"] and epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    tb_writer.close()
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    main(cfg, args.model_name)

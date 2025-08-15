import sys
import pathlib
sys.path.append(str(pathlib.Path().resolve().parent))
import numpy as np, matplotlib.pyplot as plt, yaml

from src.datasets.csf_volume_dataset import _center_crop

# --- Configuration ---
cfg = yaml.safe_load(open('config.yaml'))
data_root = pathlib.Path() / cfg['data']['root']
crop_size = cfg['data']['crop_size']
pred_dir  = pathlib.Path("outputs/preds")  # Assuming *_pred.npy are here
out_dir   = pathlib.Path("outputs/summary_images")
out_dir.mkdir(parents=True, exist_ok=True)

def generate_summary_image(dataset_name: str):
    subj_root = data_root / dataset_name
    subj_dirs = sorted([p for p in subj_root.iterdir() if p.is_dir()])
    n_subjects = len(subj_dirs)

    fig, axs = plt.subplots(nrows=n_subjects, ncols=5, figsize=(15, 3 * n_subjects))
    if n_subjects == 1:
        axs = np.expand_dims(axs, 0)  # ensure 2D for 1 subject

    for i, subj_dir in enumerate(subj_dirs):
        subj_id = subj_dir.name
        phase = np.load(subj_dir / 'phase.npy')
        mask  = np.load(subj_dir / 'mask.npy')

        phase0 = phase[0]
        phase_crop = _center_crop(phase0, crop_size) / max(np.abs(phase0).max(), 1e-5)
        mask_crop = _center_crop(mask, crop_size)

        # Load predicted mask if exists
        pred_mask_path = pred_dir / f"{subj_id}_pred.npy"
        if pred_mask_path.exists():
            pred_mask = np.load(pred_mask_path)
            if pred_mask.shape != phase_crop.shape:
                pred_mask_crop = _center_crop(pred_mask, crop_size)
            else:
                pred_mask_crop = pred_mask
            pred_mask_bin = (pred_mask_crop > 0.5).astype(np.uint8)
        else:
            pred_mask_bin = np.zeros_like(mask_crop)

        # --- 1. Raw phase[0]
        axs[i, 0].imshow(phase0, cmap='gray')
        axs[i, 0].set_title(f'{subj_id} – Phase[0]')
        axs[i, 0].axis('off')

        # --- 2. Cropped phase
        axs[i, 1].imshow(phase_crop, cmap='gray')
        axs[i, 1].set_title('Crop 64×64')
        axs[i, 1].axis('off')

        # --- 3. Manual overlay
        axs[i, 2].imshow(phase_crop, cmap='gray')
        axs[i, 2].imshow(mask_crop, cmap='Pastel1', alpha=0.4)
        axs[i, 2].set_title('Manual Overlay')
        axs[i, 2].axis('off')

        # --- 4. Manual + Predicted overlay
        overlay = np.zeros((*phase_crop.shape, 3))
        overlay[..., 0] = np.where(mask_crop > 0, 1, 0)               # Red for manual
        overlay[..., 2] = np.where(pred_mask_bin > 0, 1, 0)           # Blue for pred
        overlay[..., 1] = np.where((mask_crop > 0) & (pred_mask_bin > 0), 1, 0)  # Green for overlap
        axs[i, 3].imshow(overlay)
        axs[i, 3].set_title('Manual (R), Pred (B), Overlap (W)')
        axs[i, 3].axis('off')

        # --- 5. Predicted overlay
        axs[i, 4].imshow(phase_crop, cmap='gray')
        axs[i, 4].imshow(pred_mask_bin, cmap='Blues', alpha=0.4)
        axs[i, 4].set_title('Predicted Overlay')
        axs[i, 4].axis('off')

    plt.tight_layout()
    out_path = out_dir / f"{dataset_name}_summary.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {out_path.name} with {n_subjects} subjects")

# --- Generate for both train and test ---
generate_summary_image(cfg['data']['test_dir'])

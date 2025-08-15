import sys
import pathlib
sys.path.append(str(pathlib.Path().resolve().parent))
import numpy as np, matplotlib.pyplot as plt, yaml

from src.datasets.csf_volume_dataset import _center_crop

# --- Configuration ---
cfg = yaml.safe_load(open('config.yaml'))
data_root = pathlib.Path() / cfg['data']['root']
crop_size = cfg['data']['crop_size']
out_dir   = pathlib.Path("outputs/summary_images")
out_dir.mkdir(parents=True, exist_ok=True)

def generate_summary_image(dataset_name: str):
    subj_root = data_root / dataset_name
    subj_dirs = sorted([p for p in subj_root.iterdir() if p.is_dir()])
    n_subjects = len(subj_dirs)

    fig, axs = plt.subplots(nrows=n_subjects, ncols=3, figsize=(9, 3 * n_subjects))
    if n_subjects == 1:
        axs = np.expand_dims(axs, 0)  # ensure 2D for 1 subject

    for i, subj_dir in enumerate(subj_dirs):
        subj_id = subj_dir.name
        phase = np.load(subj_dir / 'phase.npy')
        mag   = np.load(subj_dir / 'mag.npy')
        mask  = np.load(subj_dir / 'mask.npy')

        mag0 = mag[0]
        mag_crop = _center_crop(mag0, crop_size) / mag0.max()
        mask_crop = _center_crop(mask, crop_size)

        overlay = np.stack([mag_crop, mag_crop, mag_crop], axis=-1)
        overlay[..., 0] = np.where(mask_crop > 0, 1.0, overlay[..., 0])

        axs[i,0].imshow(mag0, cmap='gray')
        axs[i,0].set_title(f'{subj_id} – Mag[0]')
        axs[i,0].axis('off')

        axs[i,1].imshow(mag_crop, cmap='gray')
        axs[i,1].set_title('Crop 64×64')
        axs[i,1].axis('off')

        axs[i,2].imshow(mag_crop, cmap='gray')
        axs[i,2].imshow(mask_crop, cmap='Pastel1', alpha=0.4)
        axs[i,2].set_title('Overlay')
        axs[i,2].axis('off')

    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_summary.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {dataset_name}_summary.png with {n_subjects} subjects")

# --- Generate for both train and test ---
generate_summary_image(cfg['data']['train_dir'])
generate_summary_image(cfg['data']['test_dir'])

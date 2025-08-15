from turtle import pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_simpson

__all__ = [
    "compute_flow_and_stroke_volume",
    "pad_to_full",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def pad_to_full(mask_crop: np.ndarray, crop_size: int, full_size: int = 240) -> np.ndarray:
    """Pad a square *crop_size*×*crop_size* mask to the *full_size* image centre."""
    pad = (full_size - crop_size) // 2
    full = np.zeros((full_size, full_size), dtype=mask_crop.dtype)
    full[pad : pad + crop_size, pad : pad + crop_size] = mask_crop
    return full

# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------

def compute_flow_and_stroke_volume(
    phase_vol: np.ndarray,
    mask: np.ndarray,
    metadata: dict,
    *,
    magnitude_vol: np.ndarray | None = None,
    ref_mask: np.ndarray | None = None,
    interpolate_n: int = 3201,
) -> dict:
    """Compute CSF flow curve and stroke volume from phase‑contrast MRI.

    Parameters
    ----------
    phase_vol : np.ndarray
        Phase‑difference volume of shape **(T, H, W)** scaled to **[‑1, 1]**.
    mask : np.ndarray
        Binary ROI mask of shape **(H, W)** selecting CSF pixels.
    metadata : dict
        Must contain the keys
            • **"v_enc"** (float, *mm/s*): velocity‑encoding factor.
            • **"pixel_size"** (float, *mm*): in‑plane pixel side length.
            • **"trigger_delay"** (float, *ms*): RR‑interval (cardiac period).
    magnitude_vol : np.ndarray, optional
        Magnitude volume (same shape as *phase_vol*). Only needed if you plan to
        build *ref_mask* on‑the‑fly; otherwise ignored.
    ref_mask : np.ndarray, optional
        Binary reference mask (same H×W) for baseline (static‑tissue) correction.
        If *None*, baseline correction is skipped.
    interpolate_n : int, default ``3201``
        Number of samples for cubic‑spline interpolation of the flow curve.

    Returns
    -------
    dict
        Keys::
            t             – time vector (*s*) at native temporal resolution
            flow          – uncorrected flow curve (*mm³/s*)
            flow_corr     – baseline‑corrected flow (equals *flow* if no ref)
            t_interp      – high‑resolution time vector (*s*)
            flow_interp   – interpolated flow (*mm³/s*)
            stroke_vol    – stroke volume (*mm³*)
            pos_area      – positive‑flow area (*mm³/s*)
            neg_area      – negative‑flow area (*mm³/s*)
            flow_range    – peak‑to‑trough of *flow_interp* (*mm³/s*)
    """
    # ------------------------------------------------------------------
    # Unpack & sanity‑check inputs
    # ------------------------------------------------------------------
    v_enc = float(metadata["v_enc"])
    pixel_size = float(metadata["pixel_size"])
    trigger_delay = float(metadata["trigger_delay"])  # ms → s later

    if phase_vol.ndim != 3:
        raise ValueError("phase_vol must be 3‑D (T,H,W)")
    if mask.ndim != 2:
        raise ValueError("mask must be 2‑D (H,W)")
    if not ((-1.1 <= phase_vol).all() and (phase_vol <= 1.1).all()):
        raise ValueError("phase_vol must be scaled to the [‑1,1] range")

    # ------------------------------------------------------------------
    # Velocity volume & mean velocity per frame
    # ------------------------------------------------------------------
    V_i = mask * phase_vol * v_enc  # (T,H,W)
    num_pixels_roi = mask.sum()
    if num_pixels_roi == 0:
        raise ValueError("The ROI mask is empty → cannot compute flow.")

    V_i_mean = V_i.sum(axis=(1, 2)) / num_pixels_roi  # (T,)

    # ------------------------------------------------------------------
    # Instantaneous flow (mm³/s) = mean velocity × ROI surface
    # ------------------------------------------------------------------
    pixel_area = pixel_size**2  # mm²
    roi_area = num_pixels_roi * pixel_area  # mm²
    flow = V_i_mean * roi_area  # mm³/s

    # ------------------------------------------------------------------
    # Static‑tissue baseline correction (optional)
    # ------------------------------------------------------------------
    flow_corr = flow
    if ref_mask is not None:
        if ref_mask.shape != mask.shape:
            raise ValueError("ref_mask must have the same spatial shape as mask")
        V_ref = ref_mask * phase_vol * v_enc
        V_ref_mean = V_ref.sum(axis=(1, 2)) / ref_mask.sum()
        flow_corr = (V_i_mean - V_ref_mean) * roi_area

    # ------------------------------------------------------------------
    # Time axis (native & interpolated)
    # ------------------------------------------------------------------
    heart_period = trigger_delay / 1e3  # s
    t = np.linspace(0, heart_period, phase_vol.shape[0], endpoint=False)

    # Cubic‑spline interpolation to refine temporal resolution
    cs = CubicSpline(np.arange(len(flow_corr)), flow_corr)
    x_new = np.linspace(0, len(flow_corr) - 1, interpolate_n)
    t_interp = np.interp(x_new / (len(flow_corr) - 1), [0, 1], [0, heart_period])
    flow_interp = cs(x_new)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    pos_area = np.sum(flow_interp[flow_interp > 0])
    neg_area = np.sum(flow_interp[flow_interp < 0])
    stroke_vol = pos_area - neg_area  # net forward volume per cycle
    flow_range = flow_interp.max() - flow_interp.min()

    return {
        "t": t,
        "flow": flow,
        "flow_corr": flow_corr,
        "t_interp": t_interp,
        "flow_interp": flow_interp,
        "stroke_vol": stroke_vol,
        "pos_area": pos_area,
        "neg_area": neg_area,
        "flow_range": flow_range,
    }

# -----------------------------------------------------------------------------
# Example usage & quick visual sanity check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib as _pl
    import pandas as pd
    # --------------------------------------------------------------
    # Load example subject data (adapt paths to your environment)
    # --------------------------------------------------------------
    DATA_ROOT = _pl.Path("data/test")
    SUBJECT_ID = "Patient_2202021349"
    subj_dir = DATA_ROOT / SUBJECT_ID

    phase_full = np.load(subj_dir / "phase.npy") * 2 - 1  # → [‑1,1]
    mag_full = np.load(subj_dir / "mag.npy")

    # Binary masks (prediction & ground truth)
    mask_pred = np.load("outputs/preds_single/" + SUBJECT_ID + "_pred.npy")
    mask_gt = np.load(subj_dir / "mask.npy")  # 240×240 

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    # Load metadata from CSV and convert to dict
    metadata_df = pd.read_csv("merged_metadata_file.csv")
    sample_metadata = metadata_df[metadata_df["sample"] == SUBJECT_ID]

    # Extract values and construct the metadata dictionary
    metadata = {
        "v_enc": float(sample_metadata["v_enc"].values[0]) * -10,  # mm/s
        "pixel_size": float(sample_metadata["pixel_size"].values[0]),  # mm
        "trigger_delay": float(sample_metadata["delay_trigger"].values[0][1:-1])  # ms
    }


    # --------------------------------------------------------------
    # Compute flow & stroke volume for both masks
    # --------------------------------------------------------------
    res_pred = compute_flow_and_stroke_volume(phase_full, mask_pred, metadata)
    res_gt = compute_flow_and_stroke_volume(phase_full, mask_gt, metadata)

    # --------------------------------------------------------------
    # Report & visualise
    # --------------------------------------------------------------
    print(f"Stroke volume (pred mask): {res_pred['stroke_vol']:.1f} mm³")
    print(f"Stroke volume (GT   mask): {res_gt['stroke_vol']:.1f} mm³")
    print(f"Ratio between GT and pred: {res_gt['stroke_vol'] / res_pred['stroke_vol']:.2f}")
    print(f"Ratio between pred and GT: {res_pred['stroke_vol'] / res_gt['stroke_vol']:.2f}")
    plt.figure(figsize=(10, 5))
    plt.plot(res_gt["t"], res_gt["flow_corr"], label="Ground Truth", marker="o")
    plt.plot(res_pred["t"], res_pred["flow_corr"], label="Prediction", marker="x")
    plt.title(f"CSF Flow – {SUBJECT_ID}")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mm³/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: accumulated volume over the cardiac cycle
    delta_t = metadata["trigger_delay"] / 1e3 / len(res_pred["flow_interp"])
    accumulated = np.cumsum(res_pred["flow_interp"]) * delta_t
    plt.figure(figsize=(10, 4))
    plt.plot(res_pred["t_interp"], accumulated)
    plt.title("Accumulated (integrated) corrected flow")
    plt.xlabel("Time (s)")
    plt.ylabel("Volume (mm³)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

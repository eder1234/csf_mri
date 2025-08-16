"""
Dice-based loss functions.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (expects raw logits)."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **_) -> torch.Tensor:
        """Compute Dice loss.

        Extra keyword arguments are ignored so that this loss can share the
        same call signature as other losses that may require additional
        inputs (e.g. phase volumes for flow-aware losses).
        """
        probs = torch.sigmoid(logits)
        num = 2.0 * torch.sum(probs * targets, dim=(1, 2, 3))
        den = torch.sum(probs + targets, dim=(1, 2, 3))
        dice = (num + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()


class FlowDiceLoss(DiceLoss):
    """Dice loss combined with a flow-consistency term.

    The flow term penalises differences between the CSF flow curves derived
    from the predicted mask and the ground-truth mask. The flow at each
    temporal frame *t* is computed as::

        flow_t = v_enc * pixel_size**2 * sum(mask * phase_vol[t])

    Parameters
    ----------
    lambda_flow : float, default ``0.1``
        Weight applied to the flow-consistency term.
    eps : float, default ``1e-5``
        Numerical stability term for the Dice component.
    """

    def __init__(self, lambda_flow: float = 0.1, eps: float = 1e-5):
        super().__init__(eps)
        self.lambda_flow = float(lambda_flow)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        phase: torch.Tensor | None = None,
        v_enc: torch.Tensor | float | None = None,
        pixel_size: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        dice_loss = super().forward(logits, targets)

        if phase is None or v_enc is None or pixel_size is None:
            return dice_loss

        # Ensure tensors on correct device and dtype
        device = logits.device
        dtype = logits.dtype
        phase = phase.to(device=device, dtype=dtype)
        v_enc = torch.as_tensor(v_enc, device=device, dtype=dtype).view(-1, 1)
        pixel_size = torch.as_tensor(pixel_size, device=device, dtype=dtype).view(-1, 1)

        probs = torch.sigmoid(logits).squeeze(1)  # (B,H,W)
        targets = targets.squeeze(1)

        # Expand masks along temporal dimension
        pred_mask = probs.unsqueeze(1)   # (B,1,H,W)
        gt_mask = targets.unsqueeze(1)

        # Compute flow curves for prediction and ground truth
        pixel_area = pixel_size ** 2
        flow_pred = v_enc * pixel_area * torch.sum(phase * pred_mask, dim=(2, 3))
        flow_true = v_enc * pixel_area * torch.sum(phase * gt_mask, dim=(2, 3))
        flow_loss = torch.mean((flow_pred - flow_true) ** 2)

        return dice_loss + self.lambda_flow * flow_loss

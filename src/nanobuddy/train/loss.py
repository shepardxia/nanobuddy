"""BiasWeightedLoss — asymmetric binary loss for wake word training."""

import torch


def bias_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_bias: float,
    clip_prob: float = 0.02,
    clip_temperature: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bias-weighted binary cross-entropy with smooth gating.

    Args:
        logits: Raw model output (pre-sigmoid).
        labels: Binary targets (0 or 1).
        loss_bias: Weight toward negative class (0..1). Higher = more negative weight.
        clip_prob: Confidence threshold for gating.
        clip_temperature: Sharpness of the sigmoid gate.

    Returns:
        (total_loss, per_example_loss) — scalar loss and detached per-example values.
    """
    yp = torch.sigmoid(logits)

    eps = 1e-7
    pos_term = -labels * torch.log(torch.clamp(yp, min=eps))
    neg_term = -(1 - labels) * torch.log(torch.clamp(1 - yp, min=eps))

    # Smooth sigmoid gate: taper loss for confident predictions
    pos_term = pos_term * torch.sigmoid(clip_temperature * ((1.0 - clip_prob) - yp))
    neg_term = neg_term * torch.sigmoid(clip_temperature * (yp - clip_prob))

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_mean = pos_term[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=logits.device)
    neg_mean = neg_term[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=logits.device)

    total = loss_bias * neg_mean + (1.0 - loss_bias) * pos_mean
    per_example = (loss_bias * neg_term + (1.0 - loss_bias) * pos_term).detach()

    return total, per_example

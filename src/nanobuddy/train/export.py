"""ONNX export with sigmoid inference wrapper."""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _InferenceWrapper(nn.Module):
    """Wraps a trained model to apply sigmoid and standardize output shape."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.trained_model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.trained_model(x)
        return torch.sigmoid(logits).view(-1, 1, 1)


def export_onnx(
    model: nn.Module,
    input_shape: tuple[int, ...],
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export a trained model to inference-ready ONNX format.

    Applies sigmoid, forces CPU, standardizes output to [batch, 1, 1].

    Returns:
        Path to the exported .onnx file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _InferenceWrapper(model).cpu().eval()
    dummy = torch.randn(1, *input_shape, device="cpu", dtype=torch.float32)

    logger.info("Exporting ONNX to %s (opset %d)", output_path, opset_version)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info("ONNX export complete: %s", output_path)
    return output_path

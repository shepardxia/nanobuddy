"""ONNX session creation, provider selection, and model download."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DOWNLOAD_URLS = {
    "melspectrogram.onnx": "https://github.com/arcosoph/nanowakeword/releases/download/models3/melspectrogram.onnx",
    "embedding_model.onnx": "https://github.com/arcosoph/nanowakeword/releases/download/models3/embedding_model.onnx",
    "silero_vad.onnx": "https://github.com/arcosoph/nanowakeword/releases/download/models3/silero_vad.onnx",
}


def default_providers(device_id: int | None = None) -> list:
    """Build an ordered ONNX provider list for the current platform.

    Tries accelerated providers first, always falls back to CPU.
    """
    providers: list = []
    if device_id is not None:
        providers.append(("CUDAExecutionProvider", {"device_id": str(device_id)}))
    elif sys.platform == "darwin":
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def create_session(path: str | Path, providers: list | None = None):
    """Create a single-threaded ONNX InferenceSession."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=providers or default_providers()
    )


def ensure_model(name: str, target_dir: str | Path) -> Path:
    """Ensure an ONNX model file exists locally, downloading if needed.

    Returns the path to the local model file.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name

    if path.exists():
        return path

    url = _DOWNLOAD_URLS.get(name)
    if url is None:
        raise FileNotFoundError(f"Unknown model '{name}' and no download URL registered.")

    logger.info("Downloading model '%s' from %s", name, url)
    _download(url, path)
    return path


def _download(url: str, target: Path) -> None:
    """Download a file from a URL."""
    import urllib.request

    tmp = target.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

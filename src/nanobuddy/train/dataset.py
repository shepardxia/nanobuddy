"""Memory-mapped dataset with hardness-aware curriculum sampling."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class HardnessDataset(Dataset):
    """Memory-mapped feature dataset with per-sample hardness tracking.

    ``feature_manifests`` is a dict of category → {key: npy_path}, e.g.::

        {"targets": {"t": "pos.npy"}, "negatives": {"n": "neg.npy"}, "backgrounds": {"b": "bg.npy"}}

    Labels: ``targets`` → 1.0, everything else → 0.0.
    """

    def __init__(self, feature_manifests: dict[str, dict[str, str]]):
        super().__init__()
        self.memmaps: list[np.memmap] = []
        self.source_info: list[dict[str, Any]] = []
        self.index_pools: dict[str, torch.Tensor] = {}

        offset = 0
        for category, manifest in feature_manifests.items():
            if not manifest:
                continue
            label = 1.0 if category == "targets" else 0.0
            for key, path in manifest.items():
                if not path:
                    continue
                try:
                    mm = np.load(path, mmap_mode="r")
                    n = len(mm)
                    self.memmaps.append(mm)
                    self.source_info.append({"label": label, "length": n, "start": offset})
                    self.index_pools[key] = torch.arange(offset, offset + n, dtype=torch.long)
                    offset += n
                except Exception as e:
                    logger.warning("Could not load '%s': %s", key, e)

        self.total_samples = offset
        self.sample_hardness = torch.ones(self.total_samples, dtype=torch.float32)
        logger.info("Dataset: %d sources, %d samples", len(self.index_pools), self.total_samples)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int):
        for info in self.source_info:
            if info["start"] <= index < info["start"] + info["length"]:
                local = index - info["start"]
                feat = self.memmaps[self.source_info.index(info)][local]
                return (
                    torch.from_numpy(feat.astype(np.float32)),
                    torch.tensor(info["label"], dtype=torch.float32),
                    index,
                )
        raise IndexError(index)


class CurriculumSampler(Sampler):
    """Hardness-weighted batch sampler with class-aware composition.

    ``batch_composition``: ``{key_or_category: count}`` — how many samples of each
    type per batch (e.g. ``{"targets": 32, "negatives": 32}``).
    """

    def __init__(
        self,
        dataset: HardnessDataset,
        batch_composition: dict[str, int],
        feature_manifests: dict[str, dict[str, str]],
        smoothing: float = 0.75,
    ):
        self.dataset = dataset
        self.composition = batch_composition
        self.manifests = feature_manifests
        self.smoothing = smoothing
        self._batch_size = sum(batch_composition.values())
        self._n_batches = self._calc_n_batches()

    def _calc_n_batches(self) -> int:
        minimum = float("inf")
        for key, quota in self.composition.items():
            if quota == 0:
                continue
            pool_size = self._pool_size(key)
            if pool_size == 0:
                return 0
            minimum = min(minimum, pool_size // quota)
        return 0 if minimum == float("inf") else int(minimum)

    def _pool_size(self, key: str) -> int:
        if key in self.dataset.index_pools:
            return len(self.dataset.index_pools[key])
        return sum(len(self.dataset.index_pools.get(k, [])) for k in self.manifests.get(key, {}))

    def _pool_indices(self, key: str) -> torch.Tensor:
        if key in self.dataset.index_pools:
            return self.dataset.index_pools[key]
        keys = list(self.manifests.get(key, {}).keys())
        return torch.cat([self.dataset.index_pools[k] for k in keys if k in self.dataset.index_pools])

    def __iter__(self):
        h = self.dataset.sample_hardness
        for _ in range(self._n_batches):
            parts = []
            for key, n in self.composition.items():
                if n == 0:
                    continue
                idx = self._pool_indices(key)
                w = (h[idx] ** self.smoothing) + 1e-6
                sel = idx[torch.multinomial(w, n, replacement=True)]
                parts.append(sel)
            if parts:
                batch = torch.cat(parts)
                yield batch[torch.randperm(len(batch))].tolist()

    def __len__(self) -> int:
        return self._n_batches


class ValidationDataset(Dataset):
    """Simple on-the-fly validation dataset from feature manifests."""

    def __init__(self, feature_manifest: dict[str, dict[str, str]]):
        super().__init__()
        self._paths: list[str] = []
        self._indices: list[int] = []
        self._labels: list[float] = []

        for category, manifest in feature_manifest.items():
            label = 1.0 if category == "targets" else 0.0
            for path in manifest.values():
                try:
                    data = np.load(path, mmap_mode="r")
                    for i in range(len(data)):
                        self._paths.append(path)
                        self._indices.append(i)
                        self._labels.append(label)
                except Exception as e:
                    logger.warning("Validation file error: %s", e)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int):
        data = np.load(self._paths[index], mmap_mode="r")
        feat = data[self._indices[index]]
        return (
            torch.from_numpy(feat.astype(np.float32)),
            torch.tensor(self._labels[index], dtype=torch.float32),
            index,
        )

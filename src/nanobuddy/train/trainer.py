"""Training loop: optimizer, scheduler, validation, checkpointing, ONNX export."""

from __future__ import annotations

import copy
import itertools
import logging
import random
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from nanobuddy.train.architectures import ARCHITECTURES
from nanobuddy.train.config import Config
from nanobuddy.train.export import export_onnx
from nanobuddy.train.loss import bias_weighted_loss

logger = logging.getLogger(__name__)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """End-to-end training for a wake word model.

    Builds the architecture from config, adds a classifier head, and runs
    the full training loop with validation, checkpointing, early stopping,
    and ONNX export.

    Example::

        from nanobuddy.train import Trainer, load_config
        config = load_config("config.yaml", overrides=["training.steps=20000"])
        trainer = Trainer(config)
        trainer.run(train_loader, val_loader)
    """

    def __init__(self, config: Config, device: str | None = None):
        _set_seed(config.training.seed)

        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Build model from architecture registry
        arch_name = config.model.architecture.lower()
        if arch_name not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{arch_name}'. Available: {list(ARCHITECTURES.keys())}")

        arch_cfg = {"embedding_dim": config.model.embedding_dim, "dropout": config.model.dropout}
        arch_cfg.update(config.model.extra)

        self.model = ARCHITECTURES[arch_name](config.model.n_features, arch_cfg)
        self.classifier = nn.Linear(config.model.embedding_dim, 1)

        self.model.to(self.device)
        self.classifier.to(self.device)

        self.input_shape = (config.model.n_features, 96)
        self.history: dict = {"loss": [], "val_loss": [], "val_loss_steps": []}

        self._setup_optimizer()

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------

    def _setup_optimizer(self):
        cfg = self.config.training
        params = list(self.model.parameters()) + list(self.classifier.parameters())

        opt_type = cfg.optimizer_type.lower()
        if opt_type == "adam":
            self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate_max, weight_decay=cfg.weight_decay)
        elif opt_type == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=cfg.learning_rate_max, momentum=0.9, weight_decay=cfg.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate_max, weight_decay=cfg.weight_decay)

        sched = cfg.lr_scheduler_type.lower()
        if sched == "cyclic":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=cfg.learning_rate_base, max_lr=cfg.learning_rate_max,
                step_size_up=int(cfg.steps * 0.4 / 3), step_size_down=int(cfg.steps * 0.6 / 3),
                mode="triangular2", cycle_momentum=False,
            )
        elif sched == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.steps, eta_min=cfg.learning_rate_base,
            )
        else:  # onecycle
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=cfg.learning_rate_max, total_steps=cfg.steps,
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, val_loader) -> dict:
        """Run validation and return metrics dict."""
        self.model.eval()
        self.classifier.eval()

        all_logits, all_labels = [], []
        with torch.no_grad():
            for features, labels, _ in val_loader:
                features = features.to(self.device)
                logits = self.classifier(self.model(features)).view(-1)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels).float()

        val_loss = F.binary_cross_entropy_with_logits(logits, labels).item()
        preds = (logits >= 0.0).float()

        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        self.model.train()
        self.classifier.train()

        return OrderedDict(
            val_loss=val_loss,
            recall=tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            fpr=fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            false_alarms=fp,
            misses=fn,
            error_score=fp + fn,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run(self, train_loader, val_loader=None, output_dir: str = "output") -> nn.Module:
        """Execute the full training loop.

        Returns the best model (eval mode, on CPU).
        """
        cfg = self.config.training
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.model.train()
        self.classifier.train()

        ema_loss = None
        best_ema = float("inf")
        no_improve = 0
        warmup = int(cfg.steps * cfg.warmup_fraction)
        patience = cfg.early_stopping_patience
        if patience is None:
            patience = int(cfg.steps * 0.15) if cfg.steps >= 3000 else 0

        best_error_score = float("inf")
        best_state = None

        # Top-k checkpoints for averaging
        top_k = 5
        top_checkpoints: list[dict] = []
        top_scores: list[dict] = []

        data_iter = iter(itertools.cycle(train_loader))

        loop = tqdm(range(cfg.steps), desc="Training")
        for step in loop:
            features, labels, indices = next(data_iter)
            features = features.to(self.device)
            labels = labels.to(self.device).float().view(-1)

            self.optimizer.zero_grad()
            emb = self.model(features)
            logits = self.classifier(emb).view(-1)

            loss, per_example = bias_weighted_loss(logits, labels, cfg.loss_bias)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.classifier.parameters()), 1.0
            )
            self.optimizer.step()
            self.scheduler.step()

            # Update hardness scores
            if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "sample_hardness"):
                ds = train_loader.dataset
                old = ds.sample_hardness[indices].to(self.device)
                ds.sample_hardness[indices] = (0.1 * per_example + 0.9 * old).cpu()

            current_loss = loss.item()
            self.history["loss"].append(current_loss)

            if ema_loss is None:
                ema_loss = current_loss
            ema_loss = cfg.ema_alpha * current_loss + (1 - cfg.ema_alpha) * ema_loss

            # Top-k checkpoint tracking
            if step > warmup:
                if len(top_checkpoints) < top_k:
                    top_checkpoints.append(copy.deepcopy(self._state_dict()))
                    top_scores.append({"step": step, "loss": ema_loss})
                else:
                    worst = max(s["loss"] for s in top_scores)
                    if ema_loss < worst:
                        idx = next(i for i, s in enumerate(top_scores) if s["loss"] == worst)
                        top_checkpoints[idx] = copy.deepcopy(self._state_dict())
                        top_scores[idx] = {"step": step, "loss": ema_loss}

            # Early stopping
            if patience > 0 and ema_loss is not None:
                if ema_loss < best_ema - 0.0001:
                    best_ema = ema_loss
                    no_improve = 0
                else:
                    no_improve += 1
                if step > warmup and no_improve >= patience:
                    logger.info("Early stopping at step %d", step)
                    break

            # Validation
            if val_loader and step > warmup and step % cfg.validation_interval == 0:
                metrics = self.validate(val_loader)
                self.history["val_loss"].append(metrics["val_loss"])
                self.history["val_loss_steps"].append(step)

                if metrics["error_score"] < best_error_score:
                    best_error_score = metrics["error_score"]
                    best_state = copy.deepcopy(self._state_dict())

            # Periodic checkpoint save
            if step > 0 and step % cfg.checkpoint_interval == 0:
                ckpt_path = out / "checkpoints"
                ckpt_path.mkdir(exist_ok=True)
                torch.save(
                    {"step": step, "model": self._state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "scheduler": self.scheduler.state_dict()},
                    ckpt_path / f"checkpoint_{step}.pth",
                )
                # Keep only latest N
                ckpts = sorted(ckpt_path.glob("checkpoint_*.pth"))
                for old in ckpts[: -cfg.checkpoint_limit]:
                    old.unlink()

            loop.set_postfix(loss=f"{current_loss:.4f}", ema=f"{ema_loss:.4f}")

        # Load best model
        if best_state:
            logger.info("Loading best model (error_score=%d)", best_error_score)
            self._load_state_dict(best_state)
        self.model.eval()
        self.classifier.eval()

        # ONNX export
        onnx_path = out / f"{self.config.model_name}.onnx"
        export_onnx(self._full_model(), self.input_shape, onnx_path, self.config.onnx_opset_version)

        return self._full_model().cpu().eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "classifier": self.classifier.state_dict()}

    def _load_state_dict(self, sd: dict):
        self.model.load_state_dict(sd["model"])
        self.classifier.load_state_dict(sd["classifier"])

    def _full_model(self) -> nn.Module:
        """Combine backbone + classifier into one sequential module."""
        return nn.Sequential(self.model, self.classifier)

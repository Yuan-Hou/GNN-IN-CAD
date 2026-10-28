"""Seed fresh runs and resume training at completed epoch boundaries."""

import os
import platform
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True):
    # Set before the first CUDA operation in a fresh kernel/process.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"], numpy_state["keys"].numpy().astype(np.uint32),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume with the same CUDA device count as the saved run.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def save_training_checkpoint(path, model, optimizer, epoch, seed, config,
                             history, batch_history, scaler=None, extra=None):
    """Export weights, then atomically publish a full epoch checkpoint last.

    DataLoaders in these notebooks use num_workers=0 and the global Torch RNG.
    A separate sampler generator or worker RNG would need its own saved state.
    """
    path = Path(path)
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "config": config,
        "history": history,
        "batch_history": batch_history,
        "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": str(np.__version__)},
        "extra": extra or {},
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    _atomic_save(model.state_dict(), path.parent / "model_weights.pt")
    _atomic_save(payload, path)


def load_training_checkpoint(path, model, optimizer, expected_config, scaler=None):
    """Restore weights, optimizer, optional AMP scaler and RNG state.

    ``expected_config`` contains settings that must match; the target epoch
    count is deliberately excluded so it can be increased on resume.
    Old checkpoints lacking RNG state cannot reproduce the original trajectory.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "rng_state" not in saved:
        raise ValueError("This older checkpoint has no RNG state; exact resume is unavailable.")
    for key, value in expected_config.items():
        if saved["config"].get(key) != value:
            raise ValueError(f"Resume setting {key!r} differs from the checkpoint.")
    model.load_state_dict(saved["model_state_dict"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(saved["scaler_state_dict"])
    restore_rng_state(saved["rng_state"])
    return saved

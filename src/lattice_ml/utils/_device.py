# Created by Javad Komijani, 2026

"""Device-selection utilities."""

import torch


__all__ = ["resolve_device"]


# =============================================================================
def resolve_device(preferred_device: str = "gpu") -> str:
    """Resolve a device preference to an actual device.

    Returns the default device if `preferred_device == "default"`. If
    `preferred_device == "cuda"`, uses `cuda` if available, otherwise falls
    back to `cpu` (mps/xpu are not considered). Otherwise, checks GPU
    accelerators in the order `cuda` > `mps` > `xpu`, falling back to `cpu`
    if none are available (or if `preferred_device is "cpu"`).

    Args:
        preferred_device (str): `"cpu"` forces CPU; `"gpu"` tries
            accelerators in the above order; `"cuda"` uses cuda if available
            and otherwise falls back to cpu; `"default"` returns the default.

    Returns:
        str: One of `"cuda"`, `"mps"`, `"xpu"`, `"cpu"`, or the dafault device.
    """
    if preferred_device not in ("cpu", "gpu", "cuda", "default"):
        raise ValueError(
            f"preferred_device must be 'cpu', 'gpu', 'cuda', or 'default'; "
            f"got {preferred_device!r}."
        )
    if preferred_device == "default":
        return torch.get_default_device()
    if preferred_device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    use_accelerator = preferred_device != "cpu"
    if use_accelerator and torch.cuda.is_available():
        return "cuda"
    if use_accelerator and torch.backends.mps.is_available():
        return "mps"
    if use_accelerator and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"

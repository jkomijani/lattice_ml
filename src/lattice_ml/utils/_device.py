# Created by Javad Komijani, 2026

"""Device-selection utilities."""

import torch


__all__ = ["get_device"]


# =============================================================================
def get_device(preferred_device: str = "auto") -> str:
    """Determine which device to use.

    Checks GPU accelerators in the order `cuda` > `mps` > `xpu`, falling back
    to `cpu` if none are available (or if `preferred_device == "cpu"`).

    Args:
        preferred_device (str): `"cpu"` forces CPU; `"auto"` or `"gpu"` tries
            accelerators in the above order.

    Returns:
        str: One of `"cuda"`, `"mps"`, `"xpu"`, or `"cpu"`.

    Raises:
        ValueError: If `preferred_device` is not `"auto"`, `"cpu"`, or `"gpu"`.
    """
    if preferred_device not in ("auto", "cpu", "gpu"):
        raise ValueError(
            f"preferred_device must be 'auto', 'cpu', or 'gpu'; "
            f"got {preferred_device!r}."
        )
    use_accelerator = preferred_device != "cpu"
    if use_accelerator and torch.cuda.is_available():
        return "cuda"
    if use_accelerator and torch.backends.mps.is_available():
        return "mps"
    if use_accelerator and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"

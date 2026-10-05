# Copyright (c) 2023-2026 Javad Komijani

"""Some generic functions mainly for tests."""

import torch


def get_machine_precision():
    """Get machine precision."""
    return 2**(-52 if torch.get_default_dtype() == torch.float64 else -23)


def get_default_tolerance():
    """Get default tolerance."""
    return 2**(-46 if torch.get_default_dtype() == torch.float64 else -18)


def eye_like(x: torch.Tensor) -> torch.Tensor:
    """
    Return identity matrices matching x's shape, dtype, and device.
    The last two dimensions of x must be square.
    """
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return eye.repeat(*x.shape[:-2], 1, 1)


def sort_eig(eigvals, eigvecs, descending=False, sort_by="real"):
    """Sort a batch of eigenvalues and permute the corresponding eigenvectors.

    Args:
        eigvals (Tensor): (..., n) tensor of eigenvalues.
        eigvecs (Tensor): (..., n, n) tensor of eigenvectors (as columns),
            matching `eigvals` along the last dimension.
        descending (bool): sorting order. (Default is False, i.e. ascending.)
        sort_by (str): key used for sorting. "real" sorts by the real part
            (the right choice for real eigenvalues, and the default); "angle"
            sorts by `torch.angle`, i.e. the complex phase, which is
            meaningful for eigenvalues of unitary matrices (all of unit
            modulus) and disambiguates values that share a real part.

    Returns:
        Tensor, Tensor: the sorted eigenvalues and correspondingly permuted
        eigenvectors.
    """
    if sort_by == "real":
        key = eigvals.real
    elif sort_by == "angle":
        key = torch.angle(eigvals)
    else:
        raise ValueError(f"sort_by must be 'real' or 'angle', got {sort_by!r}")

    _, sorted_ind = torch.sort(key, dim=-1, descending=descending)
    eigvals = eigvals.gather(-1, sorted_ind)

    n = eigvecs.shape[-1]
    sorted_ind = sorted_ind.unsqueeze(-2).repeat(*[1]*(eigvecs.ndim - 2), n, 1)
    eigvecs = eigvecs.gather(-1, sorted_ind)
    return eigvals, eigvecs


def fix_phase(eigvecs, max_row=True, first_row=False):
    """Fix the arbitrary phase of each eigenvector.

    When `max_row` is True, for each eigenvector, the element with largest
    absolute value is changed to become real.

    For a behavior similar to pytorch set `max_row = False, first_row = True`.
    """
    if max_row:
        ind_max = torch.max(torch.abs(eigvecs), dim=-2)[1]
        angle = torch.angle(eigvecs.gather(-2, ind_max.unsqueeze(-2)))
    elif first_row:
        angle = torch.angle(eigvecs[..., 0, :]).unsqueeze(-2)
    else:
        raise ValueError("Either 'max_row' or 'first_row' must be True.")

    phasor = torch.exp(-1j * angle)
    n = eigvecs.shape[-1]
    return eigvecs * phasor.repeat(*[1]*(eigvecs.ndim - 2), n, 1)


def eigvecs_accuracychecker(matrix, eig_func, return_error_norm=True, **kwargs):
    """Check accuracy of eigenvector determination."""
    u, v = eig_func(matrix, **kwargs)
    null_error = matrix @ v - v @ (torch.diag_embed(u) + 0j)
    unitary_error = v.adjoint() @ v - eye_like(v)
    if return_error_norm:
        null_error = torch.linalg.matrix_norm(null_error)
        unitary_error = torch.linalg.matrix_norm(unitary_error)
    return null_error.numpy(), unitary_error.numpy()

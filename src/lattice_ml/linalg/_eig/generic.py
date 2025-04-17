# Copyright (c) 2023 Javad Komijani

import torch


def get_machine_precision():
    return 2**(-52 if torch.get_default_dtype() == torch.float64 else -23)
    

def get_default_tolerance():
    return 2**(-46 if torch.get_default_dtype() == torch.float64 else -18)


def eyes_like(matrix):
    eye = torch.zeros_like(matrix)
    for k in range(matrix.shape[-1]):
        eye[..., k, k] = 1
    return eye


def sort_eig(eigvals, eigvecs):
    """sort eigenvalues in ascending order."""
    eigvals, sorted_ind = torch.sort(eigvals, dim=-1)
    n = eigvecs.shape[-1]
    sorted_ind = sorted_ind.unsqueeze(-2).repeat(*[1]*(eigvecs.ndim - 2), n, 1)
    eigvecs = eigvecs.gather(-1, sorted_ind)
    return eigvals, eigvecs


def fix_phase(eigvecs, max_row=True, firs_row=False):
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
    phasor = torch.exp(-1j * angle)
    n = eigvecs.shape[-1]
    return eigvecs * phasor.repeat(*[1]*(eigvecs.ndim - 2), n, 1)


def eigvecs_accuracychecker(matrix, eig_func, return_error_norm=True, **kwargs):
    u, v = eig_func(matrix, **kwargs)
    null_error = matrix @ v - v @ (torch.diag_embed(u) + 0j)
    unitary_error = v.adjoint() @ v - eyes_like(v)
    if return_error_norm:
        null_error = torch.linalg.matrix_norm(null_error)
        unitary_error = torch.linalg.matrix_norm(unitary_error)
    return null_error.numpy(), unitary_error.numpy()

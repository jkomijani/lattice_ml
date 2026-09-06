# Copyright (c) 2023-2025 Javad Komijani

# 13/May/2025: Thanks to Jarl for pointing out that QR decomposition is slow on
# GPUs

"""This module has has wrappers to `torch.linalg.qr`."""

import torch


# =============================================================================
def haar_qr(
    x: torch.Tensor,
    q_only: bool = False,
    makesure_performed_on_cpu: bool = True
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Return a phase-corrected QR decomposition suitable for generating unitary
    matrices distributed according to the Haar measure.

    This function adjusts the QR decomposition of a matrix so that the diagonal
    entries of the `r` matrix are real and positive. This phase correction
    ensures that the resulting `q` matrix is uniformly distributed with respect
    to the Haar measure on the unitary group.

    If `makesure_performed_on_cpu` is True, the decomposition will be performed
    on the CPU (via `qr_on_cpu`) and results will be moved back to the original
    device. Note that the performance of QR decomposition seems to be slower on
    available GPUs compared to CPUs (by a factor of 10 for 3 x 3 matrices with
    batch size of 64 x 1024).

    For details, see:
    [Mezzadri] F. Mezzadri, "How to generate random matrices from the
    classical compact groups", arXiv:math-ph/0609050.

    Args:
        x (torch.Tensor): A 2D or batched 2D tensor for QR decomposition.

        q_only (bool, optional): If True, only the `q` matrix is returned.
            If False (default), both `q` and `r` are returned.

        makesure_performed_on_cpu (bool, optional): If True, the QR
            decomposition is forced to run on the CPU to avoid GPU-specific
            issues, e.g. slow performance. Default is True.

    Returns:
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]:
            The orthogonal matrix `q`, and optionally the upper-triangular
            matrix `r` with phase correction such that `x = q @ r`.
    """
    if makesure_performed_on_cpu:
        q, r = qr_on_cpu(x)
    else:
        q, r = torch.linalg.qr(x, mode='complete')

    # Extract diagonal elements of r and compute their phases
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    phase = d / torch.abs(d)

    # Apply phase correction to q (columns) and r (rows)
    q = q * phase.unsqueeze(-2)
    if q_only:
        return q

    r = r * (1 / phase).unsqueeze(-1)

    # Note: x = q @ r remains valid after phase correction
    return q, r


# =============================================================================
def haar_sqr(
    x: torch.Tensor,
    u1_matrix_rep: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Decompose a matrix into SU(n), U(1), and upper-triangular components.

    This function performs a "special QR" (SQR) decomposition of a complex
    square matrix `x`, where the unitary matrix `q` from QR is further
    decomposed into a special unitary matrix `su` (with determinant 1)
    and a U(1) phase factor `u`.

    The decomposition is as follows:
        - `x = q @ r`
        - `q = su * u`
        - Thus, `x = (su * u) @ r`

    The U(1) component `u` can be returned either as a complex scalar (default)
    or as a matrix of shape `(n, n)` proportional to identity defined as:
    :math:`u = \det(q)^{1/n} \cdot I_{n \times n}`.

    Args:
        x (torch.Tensor): A complex square matrix (or batch thereof) to
            decompose via the special QR method.

        u1_matrix_rep (bool, optional): If False (default), return `u`
            as a complex scalar. If True, return a matrix representation
            of the same shape as `q`.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple `(su, u, r)` where:
              - `su`: special unitary matrix (det(su) = 1)
              - `u` : U(1) element, either scalar or matrix
              - `r` : upper triangular matrix from QR
    """
    q, r = haar_qr(x)

    # Compute the determinant of q
    det = torch.linalg.det(q)  # det \in U(1)
    # Addtional remarks of `det`
    # 2. `det`^(1/n) covers only 1/n-th of U(1) -> volume 1/n-th of U(1)
    # 3. determinant of `det`^(1/n) I_{n x n} covers U(1); stretching fac. n

    # Normalize q to have determinant 1 (i.e., into SU(n))
    n = x.shape[-1]
    phase = torch.pow(det, -1.0 / n).unsqueeze(-1).unsqueeze(-1)

    # If matrix representation of U(1) is requested
    if u1_matrix_rep:
        # u = (det(q)^{1/n}) * I, returned as a matrix
        u = torch.diag_embed(
            torch.repeat_interleave(1 / phase.squeeze(-1), n, dim=-1)
        )
    else:
        # u returned as a complex scalar
        u = det

    # su = q * phase correction => det(su) = 1
    return q * phase, u, r


# =============================================================================
def qr_on_cpu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs QR decomposition of a tensor on the CPU and returns the result on
    the original device.

    This function moves the input tensor `x` to the CPU, computes its
    QR decomposition using 'complete' mode (producing full-sized orthogonal and
    upper triangular matrices), and then moves the results back to the original
    device of `x`.

    Args:
        x (torch.Tensor): A 2D or batched 2D tensor to decompose. It can reside
           on any device (CPU or CUDA).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple `(q, r)` where:
            - `q` is an orthogonal matrix with shape matching `x` in the first
            dimension and possibly larger in the second.
            - `r` is an upper triangular matrix such that `x = q @ r`.
        Both tensors are returned on the same device as the input.
    """
    # Move the input tensor to CPU for QR decomposition
    q, r = torch.linalg.qr(x.to('cpu'), mode='complete')

    # Move the results back to the original device
    return q.to(x.device), r.to(x.device)

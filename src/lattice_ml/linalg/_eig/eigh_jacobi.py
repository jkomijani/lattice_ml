# Copyright (c) 2023 Javad Komijani

import torch
import warnings

from .generic import sort_eig, fix_phase, get_default_tolerance


def jacobi_diagonalization(A,
        max_iterations=100, tol=None, print_details=False
        ):
    r"""
    The Jacobi method is a method for the diagonalization of a hermitian matrix
    :math:`A`. It is an iterative method that uses unitary transformations
    on :math:`2\times 2` subspaces and suppress the off-diagonal elements.
    The transformation matrix :math:`P` differs
    from the unit matrix only in 4 elemenets as:

    :raw-latex:`\begin{align}
        P[p, p] &= \cos \theta \\
        P[p, q] &= e^{-i\varphi} \sin \theta \\
        P[q, p] &= -e^{i\varphi} \sin \theta \\
        P[q, q] &= \cos \theta
    \end{align}`

    For real matrices, the complex phase :math:`\varphi` vanishes.
    The indices :math:`(p, q)` are chosen such that :math:`|A[p, q]|`
    is the largest off-diagonal element.
    """
    if tol is None:
        tol = get_default_tolerance()

    assert A.shape[-2] == A.shape[-1], "A must be square"

    is_complex = torch.is_complex(A.ravel()[0])
    n = A.shape[-1]
    A_new = A.clone().reshape(-1, n, n)  # evolove A_new till it is diagonal
    
    P = torch.zeros_like(A_new)  # eigenvectors if the algorithm converges
    for k in range(n):
        P[:, k, k] = 1

    for k_iteration in range(max_iterations):
        A_diag = torch.diag_embed(torch.diagonal(A_new, dim1=-1, dim2=-2))
        
        # Find the indices (p, q) of the largest off-diagonal element
        off_diag_max, ind = torch.max(torch.abs(A_new - A_diag).reshape(-1, n**2), dim=-1)
        
        # Break if largest off diagonal term is alread small
        if torch.max(off_diag_max) < tol:
            break
                    
        p, q = ind // n, ind % n
        pp = (p * (n + 1)).view(-1, 1)
        qq = (q * (n + 1)).view(-1, 1)
        pq = (p * n + q).view(-1, 1)
        qp = (q * n + p).view(-1, 1)

        # Construct the rotation matrix J
        J = torch.zeros_like(A_new)
        for k in range(n):
            J[:, k, k] = 1

        A_ = A_new.reshape(-1, n**2)
        J_ = J.reshape(-1, n**2)
        
        if is_complex:
            theta = 0.5 * torch.atan2(
                                2 * torch.abs(A_.gather(1, pq)),
                                (A_.gather(1, qq) - A_.gather(1, pp)).real
                                )
            angle = torch.angle(A_.gather(1, pq))
            phasor = torch.exp(1J * angle)
            
            # scatter_ is the reverse of the manner described in gather()
            J_.scatter_(1, pp, torch.cos(theta) + 0J)
            J_.scatter_(1, qq, torch.cos(theta) + 0J)
            J_.scatter_(1, pq, torch.sin(theta) * phasor)
            J_.scatter_(1, qp, -torch.sin(theta) * phasor.conj())
        else:
            theta = 0.5 * torch.atan2(
                                2 * A_.gather(1, pq),
                                (A_.gather(1, qq) - A_.gather(1, pp))
                                )
            J_.scatter_(1, pp, torch.cos(theta))
            J_.scatter_(1, qq, torch.cos(theta))
            J_.scatter_(1, pq, torch.sin(theta))
            J_.scatter_(1, qp, -torch.sin(theta))
        
        # Update A and P by performing the similarity transformation
        A_new = J.adjoint() @ A_new @ J
        P = P @ J
    else:
        warnings.warn("Reached max_iterations in jacobi_diagonalization")

    if print_details:
        print(f"jacobi converged off_diag_max = {off_diag_max}, k_iter = {k_iteration}")
    eigvals = torch.diagonal(A_new, dim1=-2, dim2=-1).real
    eigvecs = P

    # Let us now sort the eigenvalues and choose a criterion to fix the phase.
    eigvals, eigvecs = sort_eig(eigvals, eigvecs)
    eigvecs = fix_phase(eigvecs)

    return eigvals.reshape(A.shape[:-1]), eigvecs.reshape(A.shape)

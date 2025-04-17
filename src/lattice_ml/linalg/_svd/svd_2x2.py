# Copyright (c) 2023 Javad Komijani

import torch


from .svd_general import AttributeDict


def svd_for_su2sums(matrix):
    """
    Special case where x is a sum of SU(2) matrices, for which one can show
    x.adjoint() @ x is proportional to the identity matrix.
    """
    s = (torch.abs(torch.linalg.det(matrix))**0.5).unsqueeze(-1)
    u = matrix / s.unsqueeze(-1)
    vh = torch.zeros_like(u)
    vh[..., 0, 0] = 1.
    vh[..., 1, 1] = 1.
    uvh = u @ vh
    det_uvh = torch.det(uvh)
    uvh[..., 0] = uvh[..., 0] / det_uvh.unsqueeze(-1)  # change only 1st column
    s = torch.cat([s, s], dim=-1)
    return AttributeDict(U=u, S=s, Vh=vh, det_uvh=det_uvh, sUVh=uvh)

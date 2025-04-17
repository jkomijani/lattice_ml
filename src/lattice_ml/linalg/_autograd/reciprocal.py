# Copyright (c) 2024 Javad Komijani

r"""
In this module we implement our own custom autograd Functions by subclassing
torch.autograd.Function and implementing the forward and backward passes
which operate on Tensors.

For example, one can see "An extended collection of matrix derivative results
for forward and reverse mode algorithmic differentiation."
"""

import torch


class Reciprocal(torch.autograd.Function):
    """Return the inverse of nonvanishing numbers or zero otherwise."""

    @staticmethod
    def forward(ctx, delta):
        nabla = 1 / delta
        nabla[ delta == 0 ] = 0
        ctx.save_for_backward(nabla)
        return nabla

    @staticmethod
    def backward(ctx, grad_nabla):
        # grad_{nabla} are $\bar \nabla$ in the terminology of AD
        nabla, = ctx.saved_tensors
        return - grad_nabla * nabla.conj()**2

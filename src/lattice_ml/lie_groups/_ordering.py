# Copyright (c) 2026 Javad Komijani

"""General-purpose ordering utilities: `torch.sort` wrapped with a
reversible index (`Order`), and `ZeroSumOrder`, which additionally imposes
a zero-sum constraint by shifting a single designated element by one
period before sorting (see `ZeroSumOrder.zerosum`).

Copied here from `normflow.lib.matrix_handles.ordering`, which also has
`ModalOrder` -- eigenvector-based, specific to reconstructing SU(n)
matrices -- not moved here since it isn't general-purpose sorting.
"""

import torch


__all__ = [
    "Order",
    "ZeroSumOrder",
]


# =============================================================================
class Order:
    """A class to perform torch.sort, but with a method to revert the sorting.

    Creating an instance and sorting the input are integrated.

    Once an instance is created, the argsort indices (the indices that sort
    a tensor along a given dimension in ascending order by value) are obtained
    and saved along with the sorted values (as properties `sorted_ind` and
    `sorted_val`, respectively).

    Use the `sort` method to sort a new tensor with respect to `sorted_ind`.
    Use the `revert` method to revert the sorting operation.
    """
    def __init__(self, x, dim=-1):
        self.dim = dim
        self.sorted_val, self.sorted_ind = self._sort(x, dim=dim)

    def _sort(self, x, dim=-1):
        return x.sort(dim=dim)

    def sort(self, x):
        """Sort (gather) x according to self.sorted_ind."""
        return x.gather(self.dim, self.sorted_ind)

    def revert(self, x):
        """Revert the sorting (gathering) operation."""
        return x.gather(self.dim, torch.argsort(self.sorted_ind))


# =============================================================================
class ZeroSumOrder(Order):
    """A class to perform torch.sort but after imposing the zero sum condition.

    For details of initiating, sorting, and reverting see `Order`.
    For details of imposing zero sum condition see `self._sort`.
    """
    def _sort(self, x, dim=-1):
        """Given a tensor of phases `x`, sort the tensor after a "designated"
        element is changed to make the sum of tensor elemenets zero
        (in the `dim` dimension).

        Idially, it is assumed that the input phases `x` are in `(-pi, pi]`,
        and their sum over the axis specified by `dim` is a multiplicative of
        `2 pi`.
        No warning will be raised if the ideal assumption is not satisfied.

        We use "canonic" to denote a specific transformation of input phases.
        First, the input phases are ordered; this gives inital ordered indices.
        Second, depending on the sum of input phases on the axis specified by
        `dim`, one phase is designated to be shifted by a factor of `2 pi` so
        that the total sum vanishes. Then all phases are in `(-pi, pi]` except
        at most one "designated" phase.
        Third, after shifting the value of "designated" phase, the previously
        sorted values are sorted again.

        The method returns the final zero-sum-sorted values and indices.

        See `zerosum` method for explanation of the method of specifying the
        "designated" phase.
        """
        val, ind = x.sort(dim=dim)
        val = self.zerosum(val, dim=dim)
        val, ind_prime = val.sort(dim=dim)  # "val" needs to be sorted agian
        ind = ind.gather(dim, ind_prime)  # "ind" must be sorted accordingly
        return val, ind

    @staticmethod
    def zerosum(x, dim=-1, period=2 * torch.pi):
        """Change a "designated" element such that the sum of `x` vanishes.

        The designated element is obtained by first finding the winding number
        of the input `x`, i.e., the sum modulo the period of elements.
        If the winding number is the minimum (maximum) possible value, the
        first (last) element is the designated one, otherwise a simple linear
        map specifies the designated value.
        """
        n_w = roundint(torch.sum(x, dim=dim) / period).unsqueeze(dim)
        n_d = x.shape[-1]
        # we now determine the designated item and change its value to the
        # zero-sum end. The index of designated item is (n_w + n_d//2) % n_d
        # `n_w` is int64 and `period` a Python float, so their product is
        # always the *default* float dtype regardless of `x.dtype` (e.g.
        # float32 even if `x` is float64) -- cast explicitly to avoid a
        # dtype mismatch in `scatter_add_` under double precision.
        shift = (-period * n_w).to(x.dtype)
        x.scatter_add_(dim, (n_w + n_d // 2) % n_d, shift)
        return x


def roundint(x, dtype=torch.int64):
    """Return the closest integer to `x`."""
    x_ = torch.round(x)
    return x_ if dtype is None else x_.type(dtype)

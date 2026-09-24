# Copyright (c) 2024-2026 Javad Komijani

"""
Defines `TupleVar`, a lightweight container for elementwise algebraic
operations on a tuple of variables.
"""

import torch


__all__ = ["TupleVar"]


# =============================================================================
class TupleVar:
    """
    A lightweight container for elementwise algebraic operations on a tuple of
    variables.

    This class wraps a tuple of objects (typically tensors or scalars) and
    supports basic arithmetic operations such as addition, subtraction,
    scalar multiplication, and division. Operations are applied elementwise
    across the contained variables, making it useful for manipulating multiple
    related state variables in ODE or optimization contexts.
    """

    def __init__(self, *args):
        """
        Initializes a TupleVar from a sequence of variables.

        Args:
            *args: A sequence of variables (e.g., tensors, scalars) to store.
        """
        self.tuple = args

    def __str__(self):
        """
        Returns a string representation of the TupleVar.
        """
        return f"TupleVar:\n{self.tuple}"

    def __repr__(self):
        """
        Returns a developer-friendly string representation.
        """
        return self.__str__()

    def __iter__(self):
        """
        Iterates over the contained variables, enabling direct unpacking,
        e.g. ``var, grad_var = aug_var``.
        """
        return iter(self.tuple)

    def __len__(self):
        """
        Returns the number of contained variables.
        """
        return len(self.tuple)

    def __getitem__(self, index):
        """
        Returns the variable at `index`.
        """
        return self.tuple[index]

    def __pos__(self):
        """
        Unary plus: returns self.
        """
        return self

    def __neg__(self):
        """
        Unary negation: returns a new TupleVar with negated elements.
        """
        return TupleVar(*[-var for var in self.tuple])

    @staticmethod
    def _pairing_of(other):
        """
        Returns the tuple to pair against for a binary op, or None if `other`
        should instead be broadcast against every contained element.
        """
        if isinstance(other, TupleVar):
            return other.tuple
        if isinstance(other, tuple):
            return other
        return None

    def __add__(self, other):
        """
        Addition (elementwise).

        If `other` is a TupleVar or a plain tuple, adds pairwise, e.g.
        `TupleVar(a, b) + TupleVar(c, d) == TupleVar(a + c, b + d)`.
        If 'other' is longer, extra elements are ignored.
        Otherwise, `other` is broadcast against every contained element.
        """
        pairing = self._pairing_of(other)
        if pairing is not None:
            x = [var1 + var2 for var1, var2 in zip(self.tuple, pairing)]
            return TupleVar(*x)
        return TupleVar(*[var + other for var in self.tuple])

    def __sub__(self, other):
        """
        Subtraction (elementwise).

        If `other` is a TupleVar or a plain tuple, subtracts pairwise, e.g.
        `TupleVar(a, b) - TupleVar(c, d) == TupleVar(a - c, b - d)`.
        If 'other' is longer, extra elements are ignored.
        Otherwise, `other` is broadcast against every contained element.
        """
        pairing = self._pairing_of(other)
        if pairing is not None:
            x = [var1 - var2 for var1, var2 in zip(self.tuple, pairing)]
            return TupleVar(*x)
        return TupleVar(*[var - other for var in self.tuple])

    def __mul__(self, other):
        """
        Multiplication (elementwise).

        If `other` is a TupleVar or a plain tuple, multiplies pairwise, e.g.
        `TupleVar(a, b) * TupleVar(c, d) == TupleVar(a * c, b * d)`.
        If 'other' is longer, extra elements are ignored.
        Otherwise, `other` is broadcast against every contained element.
        """
        pairing = self._pairing_of(other)
        if pairing is not None:
            x = [var1 * var2 for var1, var2 in zip(self.tuple, pairing)]
            return TupleVar(*x)
        return TupleVar(*[var * other for var in self.tuple])

    def __truediv__(self, other):
        """
        Division (elementwise).

        If `other` is a TupleVar or a plain tuple, divides pairwise, e.g.
        `TupleVar(a, b) / TupleVar(c, d) == TupleVar(a / c, b / d)`.
        If 'other' is longer, extra elements are ignored.
        Otherwise, `other` is broadcast against every contained element.
        """
        pairing = self._pairing_of(other)
        if pairing is not None:
            x = [var1 / var2 for var1, var2 in zip(self.tuple, pairing)]
            return TupleVar(*x)
        return TupleVar(*[var / other for var in self.tuple])

    def __rmul__(self, other):
        """
        Right-hand scalar multiplication (elementwise).
        """
        return self.__mul__(other)

    def abs(self):
        """
        Returns a new TupleVar with the absolute value of each element.
        """
        return TupleVar(*[var.abs() for var in self.tuple])

    def sum(self, **kwargs):
        """
        Returns a new TupleVar with each element reduced to its own sum.
        """
        return TupleVar(*[var.sum(**kwargs) for var in self.tuple])

    def reshape(self, *shape):
        """
        Returns a new TupleVar with each element reshaped to `shape`.
        """
        return TupleVar(*[var.reshape(*shape) for var in self.tuple])

    def cpu(self):
        """
        Returns a new TupleVar with each element moved to the CPU.
        """
        return TupleVar(*[var.cpu() for var in self.tuple])

    def numpy(self):
        """
        Returns a new TupleVar with each element converted to a numpy array.
        """
        return TupleVar(*[var.numpy() for var in self.tuple])

    @property
    def shape(self):
        """
        Returns a tuple of shapes of the contained variables.

        This property scans through each element in the tuple and retrieves
        the shape of each tensor. For non-tensor elements (including None),
        it returns None as a default shape. The returned tuple contains the
        shapes of all elements in the tuple, where each shape is either a
        tensor's shape or None for non-tensors.

        Returns:
            tuple: A tuple containing the shapes of the contained variables.
                   If an element is not a tensor, None is used as a placeholder
                   for its shape.

        Example:
            If `self.tuple = (None, torch.randn(2, 3), torch.ones(4, 5))`,
            the result of `shape` will be `(None, (2, 3), (4, 5))`.
        """
        return tuple(getattr(var, "shape", None) for var in self.tuple)

    @property
    def device(self):
        """
        Returns the device of the first non-None tensor in the tuple, or None
        if no tensors are found.

        This property scans through the tuple and identifies the device of the
        first tensor it encounters. If no tensors are present in the tuple, it
        returns None.
        """
        for var in self.tuple:
            if isinstance(var, torch.Tensor):
                return var.device

        return None

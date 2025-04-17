# Copyright (c) 2023 Javad Komijani

import torch

from .generic import fix_phase, eyes_like


# =============================================================================
def eigvals3x3(matrix, return_invariants=False):
    r"""Return eigenvalues of 3x3 matrices using a closed form expression.

    To obtain the eigenvalues, we solve the characteristic equation:

    .. math::

        \det(A - s) = - s^3 + 3 b * s^2 + 3 c * s + d

    A change of variable as :math:`s = x + b` turns the above equation to

    .. math::

        x^3 - 3 p x = 2 q

    The damped cubic equation can be solved exactly. For instance, one can
    follow Appendix A in [http://arxiv.org/abs/physics/0610206],
    "Efficient numerical diagonalization of hermitian 3x3 matrices",
    to organize the solutions.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.

    return_invariants : boolean
        in addition to the eigenvalues, return parameters mu, theta and phi;
        see the discription of algorithm below. (Default is False.)
    """

    assert matrix.shape[-2:] == (3, 3), "matrix is supposed to be 3x3"

    # We fist write det(A - s) = - s^3 + 3 b * s^2 + 3 c * s + d

    b = torch.mean(matrix.diagonal(dim1=-1, dim2=-2), dim=-1)

    c = (-matrix[..., 0, 0] * matrix[..., 1, 1]
        + matrix[..., 0, 1] * matrix[..., 1, 0]
        - matrix[..., 0, 0] * matrix[..., 2, 2]
        + matrix[..., 0, 2] * matrix[..., 2, 0]
        - matrix[..., 1, 1] * matrix[..., 2, 2]
        + matrix[..., 1, 2] * matrix[..., 2, 1]) / 3

    d = torch.det(matrix)

    p = b * b + c
    q = p * b + (b * c + d) / 2

    delta = torch.sqrt(q*q - p*p*p)

    r_1 = b 
    r_2 = (q + delta)**(1/3.)
    r_3 = p / r_2

    # We also need to take care of special cases:
    #
    # 1) `delta` close to zero:
    #    In this situation, `q^2 ~ p^3`, indicating that `|delta| ~ |q|` but
    #    with a phase that is vulnerable to numerical errors. To handle this
    #    problem, it is recommended to make the matrix traceless first and then
    #    solve the analytic formula to calculate the eigenvalues.
    #
    # 2) `r_2` close to zero:
    #    In this situation, which can happen e.g. when `p` is almost zero and
    #    `q` is negative, `r_2` tends to zero and `r_3` may start belowing up
    #    due to round-off errors in it denomenator. To handle this problem,
    #    when `|p| < 1e-16 |q|^(2/3)`, we can set `p = 0` and simply solve
    #    `x^3 = 2 q`

    cond = p.abs() <= 1e-16 * q.abs()**(2/3)
    r_2[cond] = 2**(1/3) * q[cond]**(1/3)
    r_3[cond] = 0

    w1 = (-1 + 1j * 3**0.5) / 2  # w = exp(i * 2 pi /3)
    w2 = (-1 - 1j * 3**0.5) / 2  # w = exp(i * 4 pi /3)

    eigvals = torch.stack(
               [r_1 + r_2 + r_3,
                r_1 + r_2 * w1 + r_3 * w2,
                r_1 + r_2 * w2 + r_3 * w1
               ],
               dim=-1
               )

    _, sorted_ind = torch.sort(eigvals.real, dim=-1)
    eigvals = eigvals.gather(-1, sorted_ind)

    if return_invariants:
        return eigvals, (r_1, r_2, r_3, p, q, delta)
    else:
        return eigvals


# =============================================================================
def eign3x3(matrix,
        func_4_eigvals = eigvals3x3,
        subtract_trace = True,
        method_4_nullspace = 'direct'
        ):
    """
    Return eigenvalues and eigenvectors of 3x3 normal matrices.
    
    A matrix is normal if and only if there exists a diagonal matrix of
    eigenvalues a unitary matrix of eigenvectors, meaning the eigenvectors can
    be orthogonalized.
    (In this function, one of the eigenvectors is constructed simply assuming
    the eigenvectors are orthonormal, which holds for normal matrices.)
    Example of normal matrices are Hermitian and unitary matrices.

    The eigenvalues are obtained by calling `eigvals3x3`, which uses a closed
    form expression.

    To calculate the eigenvectors, one can exploit vector cross products in 3
    dimensions as described in [http://arxiv.org/abs/physics/0610206]
    "Efficient numerical diagonalization of hermitian 3x3 matrices".
    (The method works for any normal matrices and with some modification can
    be used for any 3x3 matrices.)
    This method is pretty fast, but the error can be relatively large when the
    cross product is close to zero.
    We developed an alternative version that is about 30% slower but in general
    more accurate. For the former one, set `method='cross-product'` and for the
    latter one use `method='direct'`. The default is the latter one.

    Here are a few remarks about both methods:

    1.  Both methods are senstive to round-off errors when the condition number
        is large or when the eigenvalues are very close to each other.
        As a remedy, we consider following cases:

    2a. When there are (almost) two degenerate eigenvectors, one eigenvector
        can be constructed by cross product of the other two as suggested in
        the above paper. (This property holds for normal matrices.)
        Therefore, with sorted eigenvalues, we always construct the middle
        eigenvector by the cross product of the first and last eigenvectors.

    2b. When there are almost three degenerate eigenvectors, the matrix is
        close to a diagonal matrix. The eigenvalues might not be very precise
        (because the cancelations in the exact formula amplify the round-off
        effects) and the matrix of eigenvectors might not be precise either.
        To handle the problem, we first convert the matrix to a traceless
        matrix.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
    """
    # For benchmarking see
    # test_and_studies/eig_3x3/eig_decomposition_3x3_benchmarking.html 

    # We assume eigvals are sorted. There are three cases:
    #     1. all eigenvalues are different
    #     2. the first & second or the second & third eigenvalues are equal
    #     3. all three eigenvalues are (almost) degenerate
    # For case (3), matrix is almost proportional to unity matrix.
    # Putting case (3) aside we can assume that eigvals[1] is not equal to
    # both eigvals[0] and eigvals[2] at the same time. We therefore obtain the
    # eigvectors corresponding to eigvals[0] and eigvals[2] first and then
    # construct eigvals[1] as cross product of them.

    assert matrix.shape[-2:] == (3, 3), "matrix is supposed to be normal 3x3"

    match method_4_nullspace:
        case 'direct':
            func_4_nullspace = nullspace3x3
            func_4_ind = lambda k: [(k + 1) % 3, (k + 2) % 3, k % 3]
        case 'cross-product':
            func_4_nullspace = nullspace3x3_from_cross_product
            func_4_ind = lambda k: [k % 3, (k + 1) % 3, (k + 2) % 3]

    eigvecs = torch.zeros_like(matrix)
    eye = eyes_like(matrix)

    if subtract_trace:
        mu = torch.mean(matrix.diagonal(dim1=-1, dim2=-2), dim=-1).unsqueeze(-1)
        matrix = matrix - mu.unsqueeze(-1) * eye

    eigvals = func_4_eigvals(matrix)

    for k in [0, 2]:
        indices = func_4_ind(k)
        eigval = eigvals[..., k:k+1].unsqueeze(-1)
        eigvecs[..., k] = func_4_nullspace(
                matrix - eigval * eye,
                indices=indices
                )
    eigvecs[..., 1] = cross_product(eigvecs[..., 2], eigvecs[..., 0]).conj()

    eigvecs = fix_phase(eigvecs)

    if subtract_trace:
        return mu + eigvals, eigvecs
    else:
        return eigvals, eigvecs



# =============================================================================
def nullspace3x3(matrix, indices=[0, 1, 2], tol=1e-14):
    """Return the (right) null space for 3x3 matrices with a zero eigenvalue:

    .. math::
       
         M X = 0

    This function returns only one vector of the null space even if there are
    more than one vanishing eigenvalues.

    Assuming the input matrix has a null space, we calculate it in two steps:

    1. We transform the matrix so that the first and second columns become
    perpendicular to the third column.

    2. The first and second columns of the transformed matrix must be parallel,
    unless the third column is zero. The null space of the transformed matrix
    is a column matrix as :math:`(t; s; 0)` if the third column is not zero,
    otherwise it is :math:`(0; 0; 1)`.

    A remark on precision: if the third column is not identically zero, but
    very small, the above method may fail from round-off errors. To avoid the
    problem, we can switch to another method that uses cross-product of the
    first and second rows (yes, rows and not columns). But, we do not do it
    here because that method can fail too.
    """

    indices_flag = (0 in indices) and (1 in indices) and (2 in indices)
    assert indices_flag, "indices must be a permutation of [0, 1, 2]"

    # col_a, col_b, and col_c are distinct columns of matrix
    col_a = matrix[..., indices[0]].view(-1, 3)
    col_b = matrix[..., indices[1]].view(-1, 3)
    col_c = matrix[..., indices[2]].view(-1, 3)

    c_sq = torch.sum((col_c.conj() * col_c).real, dim=-1)
    coef_a = torch.sum(col_c.conj() * col_a, dim=-1) / c_sq
    coef_b = torch.sum(col_c.conj() * col_b, dim=-1) / c_sq
    # we assume c_sq does not vanish and later we take care of the exceptions

    # transform col_a and col_b to make them perpendicular to col_c
    col_a = col_a - coef_a.unsqueeze(-1) * col_c
    col_b = col_b - coef_b.unsqueeze(-1) * col_c
    
    # col_a and col_b are parallel since the matrix has a null space
    nullspace = orthonormal_to_parallel_vectors(
            col_a, col_b, indices=indices, tol=tol
            )
    nullspace[:, indices[2]] = -coef_a * nullspace[:, indices[0]] \
                               -coef_b * nullspace[:, indices[1]]

    nullnorm = torch.linalg.vector_norm(nullspace, dim=-1, keepdim=True)
    nullspace = nullspace / nullnorm

    cond = (c_sq.ravel() <= tol)  # c_sq = |c.c|
    if torch.sum(cond) > 0:
        nullspace[:, indices[0]][cond] = 0 
        nullspace[:, indices[1]][cond] = 0
        nullspace[:, indices[2]][cond] = 1

    return nullspace.view(*matrix.shape[:-1])


# =============================================================================
def nullspace3x3_from_cross_product(matrix, indices=[0, 1, 2], tol=1e-14):
    """Return the (right) null space for 3x3 matrices with a zero eigenvalue:

    .. math::
       
         M X = 0

    This function returns only one vector of the null space even if there are
    more than one vanishing eigenvalues.

    For determination of the nullspace of a 3x3 matrix, we follow
    "Efficient numerical diagonalization of hermitian 3x3 matrices"
    [http://arxiv.org/abs/physics/0610206],
    which uses properties of the vector cross product in 3x3 to construct
    the eigenvectors.

    The method spelled out in the mentioned paper returns the conjugate
    transposed of left eigenvectors which are equal to right eigenvectors for
    normal matrices, such as Hermitian and unitary matrices.
    Here we directly obtain the right eigenvectors, so that the method can be
    used for any matrices. However, if the cross product fails....
    """                                                                         
    a = matrix[..., indices[0], :].view(-1, 3)  # first picked row
    b = matrix[..., indices[1], :].view(-1, 3)  # second picked row

    nullspace = cross_product(a, b)

    nullnorm = torch.linalg.vector_norm(nullspace, dim=-1, keepdim=True)
    nullspace = nullspace / nullnorm

    cond = (nullnorm.ravel() <= tol)  # nullnorm = |a x b|
    if torch.sum(cond) > 0:  # if nullnorm is zero at least for one case
        # The following three command lines assume the matrix is normal
        a = matrix[..., indices[0]].view(-1, 3)  # 1st picked column (not row)
        b = matrix[..., indices[1]].view(-1, 3)  # 2nd picked column
        nullspace[cond] = orthonormal_to_parallel_vectors(
                a[cond], b[cond], indices=indices, tol=tol
                )

        # We could also switch to the direct method
        # nullspace[cond] = nullspace3x3(
        #       matrix.reshape(-1, 3, 3)[cond],
        #       indices=indices
        #       )

    return nullspace.view(*matrix.shape[:-1])


# =============================================================================
def cross_product(vec1, vec2):
    """Return cross product of three dimensional vectors vec1 & vec2
    (over the last axis).
    """
    vec3 = torch.zeros_like(vec1)
    vec3[..., 0] = (vec1[..., 1] * vec2[..., 2] - vec1[..., 2] * vec2[..., 1])
    vec3[..., 1] = (vec1[..., 2] * vec2[..., 0] - vec1[..., 0] * vec2[..., 2])
    vec3[..., 2] = (vec1[..., 0] * vec2[..., 1] - vec1[..., 1] * vec2[..., 0])
    return vec3


# =============================================================================
def orthonormal_to_parallel_vectors(vec1, vec2, indices=[0, 1, 2], tol=1e-14):
    """Return an orthornormal vector to three dimensional vectors vec1 & vec2
    that are assumed to be parallel.
    """
    x = torch.sum(vec1.conj() * vec2, dim=-1)
    y = torch.sum(vec1.conj() * vec1, dim=-1)
    z = torch.sqrt(x.conj() * x + y * y).real  # z = 0 -> x = y = 0 & vec1 = 0

    vec3 = torch.zeros_like(vec1)
    vec3[..., indices[0]] = -x / z
    vec3[..., indices[1]] = y / z

    cond = (z.real <= tol).ravel()
    if torch.sum(cond) > 0:
        vec3[..., indices[0]].view(-1)[cond] = 1
        vec3[..., indices[1]].view(-1)[cond] = 0

    return vec3

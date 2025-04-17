# Copyright (c) 2024 Javad Komijani

"""Solve an initial value problem for a system of ODEs."""

import torch


def odeint(
    func, t_span, var0, args=None, step_size=1e-3, method='RK4', loss_rate=None
):
    r"""Solve an initial value problem for a system of ODEs.

    The system of ODEs is::

         dy / dt = f(t, y; p)

    where ``t`` is the flow time of the system, ``y`` is the state variable
    (a vector describing the state of the system) at time ``t``, and ``p`` is
    a set of fixed parameters that specify the flow (dynamic) of the system.

    Parameters
    ----------
    func: callable
        Computes the derivative of vector ``y`` at time ``t``.
    t_span: a sequence
        First and second items are the initial and terminal flow times.
    var0: tensor
        Initial state variable, i.e. the initial value of vector ``y``.
    args: tuple[tensor]
        The parameters specifying the system.
    step_size: float
        The size of increament in the flow time ``t`` at each step
        (default is 0.001). The sign is determind by ``t_span``. Note that the
        given `step_size` is only tentative and depending on the initial and
        terminal flow times, the exact `step_size` may change slightly.
    method: str or callable
        The integration method as a string or a function (default is 'RK4').
    loss_rate: None or callable
        If callable, it will be treated as the integrand of a loss integral
        over time. It will be evaluated at each step, and summed over [with
        Simpson's rule if possible], and the sum will be returned as output
        along with the state variable.

    Remark::
    This code can handle any number of batch axes if ``func`` can handle them.
    """
    if hasattr(method, '__call__'):
        step = method
    elif method == 'RK4':
        step = rk4_step
    elif method == 'Euler':
        step = euler_step
    else:
        raise ValueError(f"{method} is not supported!")

    if args is None:
        args = ()
    elif not isinstance(args, tuple):
        args = (args,)

    n_grid = 1 + abs(int((t_span[1] - t_span[0]) / step_size))

    t_range = torch.linspace(*t_span, n_grid, device=var0.device)
    step_size = t_range[1] - t_range[0]  # update step_size

    var = var0

    # Start block if loss_rate is callable
    if hasattr(loss_rate, '__call__'):

        simpsons_rule = (n_grid % 2 == 1)  # for summing loss rate

        loss = loss_rate(t_range[0], var, *args)

        for ind, t in enumerate(t_range[:-1]):
            var = step(func, t, var, step_size, *args)
            dloss = loss_rate(t_range[ind + 1], var, *args)
            if simpsons_rule and ind % 2 == 0:
                loss += 4 * dloss
            else:
                loss += 2 * dloss
        loss -= dloss  # for the last `ind`
        if simpsons_rule:
            loss *= (step_size / 3)
        else:
            loss *= (step_size / 2)
        return var, loss
    # End of block if loss_rate is callable

    for t in t_range[:-1]:
        var = step(func, t, var, step_size, *args)

    return var


def euler_step(func, t, var, dt, *args):
    """Perform a single Euler step."""
    return var + func(t, var, *args) * dt


def rk4_step(func, t, var, dt, *args):
    """Perform a single Runge-Kutta-4 step."""
    half_dt = dt / 2
    k_1 = func(t, var, *args)
    k_2 = func(t + half_dt, var + half_dt * k_1, *args)
    k_3 = func(t + half_dt, var + half_dt * k_2, *args)
    k_4 = func(t + dt, var + dt * k_3, *args)
    return var + (k_1 + 2 * k_2 + 2 * k_3 + k_4) * (dt / 6)

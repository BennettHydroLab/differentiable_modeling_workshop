"""Neural-network and differentiability building blocks.

Two kinds of thing live here. First, the small network pieces every hybrid
model in the workshop needs (an MLP, and the map from unconstrained network
output onto physical parameter ranges). Second — and more important —
**smooth replacements for the non-differentiable operations** that conceptual
hydrologic models are full of.

That second group is the one people get wrong. A rainfall-runoff model is
written with `if`, `min`, `max` and hard thresholds all through it: rain
becomes snow below freezing, a bucket spills when it is full, percolation is
capped. Each of those has a zero or undefined derivative, so a gradient
arriving there is either killed or ill-defined. Replacing them with smooth
surrogates is usually what stands between "my hybrid model will not train"
and a working one.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Smooth surrogates for non-differentiable operations
# --------------------------------------------------------------------------

def smooth_threshold(x, center=0.0, sharpness: float = 5.0):
    """A differentiable stand-in for ``x > center``, returning a weight in (0, 1).

    The classic use is partitioning precipitation into rain and snow::

        snow_frac = smooth_threshold(-temp, center=-t_snow, sharpness=2.0)

    ``sharpness`` controls how abruptly the switch happens, in units of 1/x.
    Larger is closer to a step function but gives a flatter gradient away from
    the threshold — which is exactly how you strangle your own training. Start
    near 1/(the scale over which the transition is physically real): for a
    rain-snow threshold in degrees C, ``sharpness`` of 1-2 is sensible, not 50.
    """
    return torch.sigmoid(sharpness * (x - center))


def smooth_min(a, b, beta: float = 1.0):
    """Differentiable ``min(a, b)`` via the log-sum-exp softmin.

    Use for capped fluxes — percolation limited by available storage, melt
    limited by snowpack. ``beta`` sets the softness in the *units of a and b*:
    the approximation error is about ``log(2)/beta``, so for fluxes in mm/day
    a ``beta`` of 5-20 keeps the error well under a tenth of a mm while
    keeping gradients alive on both branches.
    """
    return -torch.logaddexp(-beta * torch.as_tensor(a), -beta * torch.as_tensor(b)) / beta


def smooth_max(a, b, beta: float = 1.0):
    """Differentiable ``max(a, b)``. See :func:`smooth_min` for ``beta``."""
    return torch.logaddexp(beta * torch.as_tensor(a), beta * torch.as_tensor(b)) / beta


def soft_clamp(x, lo, hi, beta: float = 1.0):
    """Differentiable ``clamp(x, lo, hi)``, built from the two softs above."""
    return smooth_min(smooth_max(x, lo, beta), hi, beta)


def smooth_relu(x, beta: float = 1.0):
    """Softplus — a differentiable ``max(x, 0)`` for enforcing non-negative stores.

    Note this is never exactly zero, so a storage built on it holds a small
    residual. If exact emptiness matters, say so in the notebook rather than
    pretending otherwise.
    """
    return torch.nn.functional.softplus(x, beta=beta)


# --------------------------------------------------------------------------
# Network pieces
# --------------------------------------------------------------------------

class MLP(nn.Module):
    """A plain multilayer perceptron.

    Deliberately boring: for the closures in this workshop, a couple of hidden
    layers of 16-32 units is plenty, and anything bigger mostly buys you a
    longer coffee break.
    """

    def __init__(self, n_in, n_out, hidden=32, depth=2, activation=nn.Tanh):
        super().__init__()
        layers, size = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(size, hidden), activation()]
            size = hidden
        layers += [nn.Linear(size, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ParamMap(nn.Module):
    """Map unconstrained real numbers onto physical parameter ranges.

    Optimizers work best on unbounded variables; hydrologic parameters have
    hard physical bounds. Squashing through a sigmoid and rescaling gives the
    optimizer a free hand while guaranteeing the physics never sees a negative
    field capacity.

    This is exactly the trick Lamichhane & Bennett use to keep an LSTM's
    output inside the published SNOW-17 parameter ranges::

        pmap  = ParamMap(lo=[50., 1.0], hi=[500., 5.0])
        theta = pmap(net(x))        # whatever the net says, theta is in range
    """

    def __init__(self, lo, hi):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))

    def forward(self, raw):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw)

    def inverse(self, theta):
        """Recover the unconstrained value — handy for warm-starting from a
        known calibrated parameter set."""
        p = ((theta - self.lo) / (self.hi - self.lo)).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))


# --------------------------------------------------------------------------
# Fixed-step ODE integration
# --------------------------------------------------------------------------

def rk4_step(f, u, t, dt):
    """One classical fourth-order Runge-Kutta step of ``du/dt = f(t, u)``.

    Every operation is a differentiable torch op, so autograd walks straight
    back through the solver. This is "discretize-then-optimize": we
    differentiate the code that took the steps. See
    ``research_brief.md`` §6 for how that differs from the adjoint.
    """
    k1 = f(t, u)
    k2 = f(t + dt / 2, u + dt / 2 * k1)
    k3 = f(t + dt / 2, u + dt / 2 * k2)
    k4 = f(t + dt, u + dt * k3)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint_fixed(f, u0, t):
    """Integrate ``du/dt = f(t, u)`` over the grid ``t`` with fixed RK4 steps.

    Returns the trajectory stacked along a new leading time axis. Fixed-step
    and simple on purpose — ``torchdiffeq`` does the adaptive, memory-efficient
    version, and notebook 02 shows why you would want it.
    """
    u, out = u0, [u0]
    for i in range(len(t) - 1):
        u = rk4_step(f, u, t[i], t[i + 1] - t[i])
        out.append(u)
    return torch.stack(out)

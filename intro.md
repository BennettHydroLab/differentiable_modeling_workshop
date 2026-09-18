# Differentiable & Hybrid Modeling for Hydrologic Systems

Welcome. Over the next two hours we are going to take a conceptual
hydrologic model apart, replace one piece of it with a neural network, and
train the whole thing end to end — physics and network together, with
gradients flowing through both.

That combination goes by several names in the literature: *differentiable
modeling*, *hybrid modeling*, *universal differential equations*. They point
at the same idea. A process-based model encodes what we already know about a
catchment: mass balance, the shape of a recession curve, the fact that snow
melts when it is warm. A neural network is good at exactly the part we
cannot write down — how conductivity varies with saturation, how a
parameter should shift between a wet year and a dry one. Differentiable
programming lets you keep the first and learn the second, in one model, from
data.

## What makes this different from calibration

You have calibrated models before. You picked an objective function, turned
a search algorithm loose on a dozen parameters, and waited. That works, and
it stops working somewhere around twenty parameters.

The move here is that the model is written so you can differentiate it. Once
you can compute $\partial \mathcal{L} / \partial \theta$ analytically, the
cost of finding good parameters stops scaling with how many there are. That
is what makes it reasonable to have a neural network — tens of thousands of
weights — sitting inside a water balance, and it is what lets you fit one
model across hundreds of catchments at once instead of one model per gauge.

## The shape of the workshop

:::{list-table}
:header-rows: 1
:widths: 12 30 58

* - Time
  - Block
  - What happens
* - 0:00
  - Setup & motivation
  - Get everyone running; see the punchline before the theory.
* - 0:10
  - Theory
  - Automatic differentiation, gradients through an ODE solver, and what a
    universal differential equation actually is.
* - 0:45
  - Applications
  - Build a hybrid rainfall-runoff model on real CAMELS basins. Learn a
    parameter that varies in time. Use the hybrid model as a diagnostic.
* - 1:35
  - Frontiers
  - What breaks, what is unsolved, and where to go next.
:::

## Before we start

Everything runs in the browser through the Colab badge at the top of each
notebook, or locally:

```bash
git clone https://github.com/BennettHydroLab/differentiable_modeling_workshop.git
cd differentiable_modeling_workshop
uv sync && uv run jupyter lab
```

Data comes from [`minicamels`](https://github.com/BennettHydroLab/minicamels):
50 CAMELS-US basins, 30 water years of Daymet forcings and USGS streamflow,
small enough to fetch in seconds.

## What you need to know already

Catchment hydrology and Python. That is genuinely it — we build the machine
learning from gradients upward, and no prior deep learning experience is
assumed.

# Measured performance — build your notebooks around these numbers

All measured in this repo's venv (torch 2.14, CPU, 1-4 threads) on a daily
conceptual model (HBV-lite: soil store + fast/slow routing, 6 parameters).
Numbers are from a *contended* dev box, so a participant's laptop should be
at least this fast, not slower.

## Cost of one gradient step through an unrolled simulation

| Simulation length | Batch size | Time per step |
|---|---|---|
| 730 days  | 1  | **0.45 s** |
| 1095 days | 1  | 0.69 s |
| 1460 days | 1  | 0.95 s |
| 1460 days | 16 | **1.03 s** |

Two things follow, and both are worth teaching out loud:

**1. Cost is linear in simulation length, ~0.65 ms per simulated day.** The
Python `for t in range(n_days)` loop and the autograd graph it builds are the
bottleneck, not the arithmetic. So: **budget by simulated days, not by epochs.**

> A 40-step training run over 1 water year + 1 year of spin-up (730 days)
> takes about **18 seconds**. That is the right size for a live demo. The same
> run over 30 water years would take 12 minutes and lose the room.

**2. Batching is nearly free.** 16 parameter sets cost 1.03 s/step against
0.95 s/step for one — a 16x increase in work for 8% more time, because the
loop overhead is paid once and the per-step ops are vectorized.

This is not just an optimization, it is the point of the regionalization
argument in `research_brief.md` §4: training one model across many basins
simultaneously costs barely more than training on one basin. Put basins (or
parameter sets, or ensemble members) on a **leading batch dimension** and loop
over time only.

```python
# state shape (n_basin,) not scalar — the time loop is the only loop
Ss = torch.full((n_basin,), 50.0)
for t in range(n_days):
    ...
```

## Forward-only is ~100x cheaper than a gradient step

400 parameter sets x 1460 days with `torch.no_grad()`: **0.1 s total.** Use
`no_grad` for any search, sampling, or evaluation sweep. Only pay for the
graph when you actually need the gradient.

## What this means for the motivating argument

Be honest in the notebooks: **at 6 parameters, random search is competitive
with gradient descent.** 400 random samples found NSE 0.56 in 0.1 s; 60
gradient steps reached NSE 0.58 in 58 s. Gradients do not win on wall clock
in low dimensions, and claiming they do will not survive contact with an
audience that calibrates models for a living.

The real argument is dimensional scaling, and it is overwhelming:

- A gradient step costs the **same** whether you have 6 parameters or 19,208
  (the size of a 64-unit LSTM head emitting 8 SNOW-17 parameters — see
  `research_brief.md` §4). Reverse-mode AD returns the whole gradient for
  about the cost of one extra forward pass.
- Random or grid search costs grow exponentially in dimension. A 30-point
  grid over 13 parameters is 30^13 ~ 1.6e19 evaluations.
- Shen et al.'s framing of the same point: finite differences over 10,000
  weights "would require 10,001 forward model evaluations."

So the honest story is **not** "gradients are faster." It is: *gradients are
the only thing that still works once the question mark moves from a handful
of numbers to a function.*

## Reference model quality

For calibration on Cowpasture River VA (`BASIN_TEMPERATE`), 3 water years of
training, the HBV-lite structure above reaches **NSE ~0.58**. That is a
reasonable, honest baseline for a 6-parameter conceptual model — do not tune
it into something implausible to make a hybrid model look better. The hybrid
comparison is more convincing when the physics baseline is fairly calibrated.

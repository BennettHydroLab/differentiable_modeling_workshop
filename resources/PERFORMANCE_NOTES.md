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

---

## Verified reference results (notebook 00)

Measured end to end with the final `workshop_utils` API, HBV-lite on
`BASIN_TEMPERATE` (Cowpasture River VA), 2 water years of training plus 1 year
of spin-up (1095 days), tested on WY2000-2002.

| Method | Cost | Train NSE |
|---|---|---|
| Random search, 500 parameter sets, `no_grad` | **0.13 s** | 0.643 |
| Adam, 40 gradient steps | **34.2 s** (0.86 s/step) | 0.635 |

Held-out test metrics for the gradient-calibrated parameters:
`NSE 0.526, KGE 0.654, logNSE 0.283, PBIAS 7.5%, RMSE 0.717`.

Calibrated parameters (sensible values, which is the point of `ParamMap`):
`FC 198.6, beta 4.56, LP 0.90, K_fast 0.322, K_slow 0.028, PERC 1.455`.

**Random search beat gradient descent here, and was 260x faster.** Say this
out loud in the notebooks. At six parameters the gradient is not worth its
price, and an audience that calibrates for a living already knows it.

### Batching across basins is free — the measurement

Training 16 basins simultaneously, each with its own parameter vector, 40
Adam steps:

| Setup | Time per step |
|---|---|
| 1 basin  | 0.86 s |
| 16 basins | **0.75 s** |

**16x the work for 0.88x the time.** (It comes out slightly *faster* than the
single-basin run because the per-step Python and autograd overhead is paid
once either way, and this box is noisy.) Median NSE across the 16 basins was
0.611.

This is the regionalization argument from `research_brief.md` §4 made
concrete, and it is the number to put in front of the room: the reason
differentiable modeling scales to continental domains is not that gradients
are fast, it is that the expensive part does not grow when you add basins.

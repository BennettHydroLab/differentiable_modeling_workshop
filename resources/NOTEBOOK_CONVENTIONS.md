# Notebook conventions

Binding for every notebook in this workshop. Three authors are writing in
parallel; these rules are what make the result read as one document.

## Audience

A practicing hydrologic modeler. They know catchment hydrology, have
calibrated a conceptual model, and write Python. They have **not**
necessarily trained a neural network. Never assume prior deep learning
vocabulary — introduce a term before you lean on it.

## The Colab bootstrap cell

Every notebook starts with exactly this cell, tagged `remove-output`:

```python
# --- Colab bootstrap: installs the workshop package on first run -----------
try:
    import workshop_utils
except ImportError:
    %pip install -q "git+https://github.com/BennettHydroLab/differentiable_modeling_workshop.git"
    import workshop_utils
```

Directly beneath it, a normal imports cell. Import from `workshop_utils`
rather than redefining helpers:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from workshop_utils import nse, kge, summary, hydrograph, set_style, COLORS
from workshop_utils.data import load_basin, split_by_water_year, to_tensors, BASIN_TEMPERATE

set_style()
torch.manual_seed(0)
```

## Notation — use these symbols everywhere

This scheme is arbitrated in `resources/research_brief.md` §1, which explains why
each choice was made and which collisions it resolves. **That brief is the source
of truth**; this table is the summary you keep open while writing.

| Symbol | Meaning | Code name |
|--------|---------|-----------|
| $\mathbf{u}(t)$ | model **state** vector (SWE, soil store, routing store) | `u` |
| $\mathbf{x}(t)$ | dynamic **forcings** (`prcp`, `tmax`, `tmin`, `srad`, `vp`) | `x` |
| $\mathbf{a}$ | static catchment **attributes** | `a` / `attrs` |
| $\boldsymbol{\theta}$ | interpretable **physical** parameters | `theta` |
| $\boldsymbol{\varphi}$ | **neural network weights** | `phi` / module params |
| $\mathrm{NN}_\varphi(\cdot)$ | the **learned closure** | `net` |
| $f(\cdot)$ | the **known physics** (RHS we keep) | `f` |
| $\mathbf{y}$, $\mathbf{y}_{\text{obs}}$ | model outputs / observations | `y`, `y_obs` |
| $\mathcal{L}$ | loss | `loss` |
| $z$ | vertical coordinate / depth | `z` |
| $\psi$ | pressure head (Richards) | `psi` |
| $\mathbf{h}, \mathbf{c}$ | LSTM hidden and cell state | `h`, `c` |
| $T$ | temperature (a forcing) — **not** terminal time | `temp` |

Three rules that are easy to get wrong:

- $\theta$ is **physical** parameters, never network weights. The whole arc of the
  workshop is "$\theta$ used to be a number you searched for; now it can be a
  *function*." Use $\varphi$ for weights.
- Never use $g$ — Shen et al. use it for both the physics and the network. Use $f$
  for retained physics.
- Terminal time is `t_end`, sequence/lookback length is `L`. $T$ is temperature.
  Richards' volumetric water content is $\theta_w$, and you must say once in prose
  that this is a different $\theta$ from the parameter vector.

### The five canonical forms

Every hybrid model in this workshop is a specialization of one of these. Say which
one you are building, by letter, in the notebook's opening.

```
du/dt = f(u, x, theta, t)                          # (0) pure physics
du/dt = f(u, x, NN_phi(x, a, u), t)                # (1) learned parameters
du/dt = f(u, x, theta, NN_phi(u, x, a), t)         # (2) learned process / closure
du/dt = f(u, x, theta, t) + NN_phi(u, x)           # (3) learned residual, inside the ODE
du/dt = NN_phi(u, x, t)                            # (4) neural ODE — no physics left
```

The discrete daily form the notebooks actually run:

```
theta_t   = g_hat( NN_phi(x_{1:t}, a) )      # sigmoid range-map onto physical bounds
u_{t+1}   = u_t + dt * f(u_t, x_t, theta_t)
y_t       = obs_operator(u_t, x_t, theta_t)
loss      = mean_t loss(y_t, y_obs_t)
```

### Re-symboling inherited code

The presenter's existing notebooks do **not** use one scheme — `theta` means NN
weights in some, a concatenated physical+neural vector in another, and volumetric
water content in a third. `research_brief.md` §1b lists every collision. **If you
lift code, re-symbol it to the table above.** Do not propagate the collision.

## Structure of a notebook

1. `# Title` — then a 2-3 sentence framing of the hydrologic question.
2. **A box stating time budget and what the reader will be able to do afterwards.** Use a MyST admonition.
3. Bootstrap + imports.
4. Content, in numbered `## N. Section` headings.
5. `## Your turn` — at least one exercise with a hidden/collapsible solution.
6. `## Takeaways` — 3-5 bullets, each a claim not a topic.
7. `## References` — cite with `` {cite}`key` `` against `references.bib`.

## MyST admonitions to use

````markdown
:::{admonition} Time check — 12 minutes
:class: tip
What you will be able to do: ...
:::

:::{admonition} Why this matters for hydrology
:class: note
:::

:::{admonition} Gotcha
:class: warning
:::

:::{dropdown} Solution
Hidden until clicked — use for exercise answers.
:::
````

## Runtime budget — hard limit

**No single cell may take longer than ~60 seconds on a laptop CPU.** The
whole notebook should execute in under 3 minutes. This is a live workshop;
a 5-minute training cell loses the room.

Ways to stay inside it:
- Train on a few water years, not 30.
- Small networks. A 2-layer MLP with 16-32 hidden units is plenty to make
  the point.
- A few hundred epochs, not thousands.
- If something genuinely needs longer, **precompute it**, commit the result
  to `assets/`, and load it — then show the training code in a cell marked
  as illustrative that is not executed at runtime.
- Always print training progress so the room sees something happening.

## Voice — settled, do not drift

Use **chapter 12's register**, with the course notebooks' structural
discipline. Concretely:

- Second person. "You might be thinking", "if you're ambitious", "notice that
  your model now has...". Not the course notebooks' impersonal "we".
- "Let's" is fine and in character.
- Grant permission to skip. Chapter 12 says things like *"most of the time it
  works"* and *"let's call this good enough and move on"*. Do that — this
  audience does not need every derivation.
- **Validate against a known answer before using a tool in anger.** This is the
  presenter's most distinctive habit and it is exactly right for an audience
  being asked to trust gradients through their own model. Check RK4 against an
  analytic solution; check an autograd gradient against a finite difference.
- Sustained analogies are welcome (optimization as topography).

From the course notebooks, keep: the `## Motivation` / `## What we'll cover`
opening, summary tables instead of bulleted recaps, explicit cross-references
to sibling notebooks by filename, and — most important — **stating your own
simplifications and runtime budget in prose**. Say out loud what you cut and why.

**American spelling** throughout (normalize, visualize, behavior). The source
corpus mixes both; we do not.

## Code style

- Comments explain *why*, not *what*. Match chapter 12's density.
- Small composable `nn.Module` classes over monolithic functions.
- Never leave a magic number unexplained.
- Prefer showing a plot over printing a number.
- `torch.manual_seed(0)` at the top so results reproduce in the room.

## Use the shared package — do not reimplement these

`workshop_utils` is built, tested, and importable. Three agents reinventing
`nse` is exactly how this set of notebooks stops being one document.

```python
from workshop_utils import (
    # metrics — work on numpy OR torch, differentiable, NaN-safe
    nse, kge, log_nse, pbias, rmse, summary,
    # data
    load_basin, split_by_water_year, to_tensors, potential_et,
    BASIN_TEMPERATE, BASIN_SNOWY, BASIN_ARID,
    # differentiable ops — see nn.py docstrings, they explain the pitfalls
    smooth_threshold, smooth_min, smooth_max, soft_clamp, smooth_relu,
    # network pieces
    MLP, ParamMap,
    # fixed-step integration
    rk4_step, odeint_fixed,
    # plotting
    hydrograph, set_style, COLORS,
)
```

Notes on the ones with sharp edges:

- `load_basin(id)` returns an xarray Dataset that **already includes** `tmean`
  and `pet`. PET is Hamon, rescaled per basin to match the CAMELS `mean_pet`
  attribute (raw Hamon underestimates it by 1.2-2.2x). Do not roll your own.
- `ParamMap(lo, hi)` is the sigmoid range-map — the same trick Lamichhane uses
  to keep an LSTM's output inside published SNOW-17 ranges. It has an
  `.inverse()` for warm-starting from a calibrated parameter set.
- The `smooth_*` functions are the fix for the single most common reason a
  hybrid model will not train. `smooth_min` keeps a gradient of 0.88 on a
  branch where `torch.minimum` gives exactly 0.0.

If you need something shared that is missing, **add it to `workshop_utils` and
tell the orchestrator**, rather than defining it locally in a notebook.

## Figures

Use `workshop_utils.plotting` (`hydrograph`, `COLORS`, `set_style`). The
palette is fixed so "observed" is the same grey and "hybrid" the same green
in every notebook. Do not introduce a second palette.

## Data

Use `minicamels` through `workshop_utils.data`. Running examples:

- `BASIN_TEMPERATE` = 02016000, Cowpasture River VA — humid, snow-free
- `BASIN_SNOWY` = 13313000, Johnson Creek ID — 74% snow fraction
- `BASIN_ARID` = 06353000, Cedar Creek ND — runoff ratio 0.04, hard

Available variables: `prcp` (mm/d), `tmax`, `tmin`, `tmean` (°C), `srad`
(W/m²), `vp` (Pa), `qobs` (mm/d). 50 basins, 1980-10-01 to 2010-09-30.
Static attributes: `lat`, `lon`, `elev_mean`, `slope_mean`, `area_km2`,
`mean_prcp`, `mean_pet`, `aridity`, `frac_snow`, `q_mean`, `runoff_ratio`,
`hfd_mean`, `baseflow_index`, `soil_depth_pelletier`, `frac_forest`,
`lai_max`.

**There is no observed SWE and no observed ET in minicamels.** Any snow or
evaporation example must therefore be either a synthetic-truth twin
experiment or trained against discharge. Do not write code that loads an
SWE observation — it does not exist.

Always split train/test in **time**, by water year, and say so out loud —
random shuffling of time steps is the most common silent mistake in this
field.

## Testing your notebook — read this before you burn an hour

Prototype your code as a `.py` script first, get it fast and correct, *then*
turn it into a notebook with `tools/build_notebook.py`. Authoring `.ipynb`
JSON by hand is not worth it.

```bash
# author from a plain-text source; split cells with `#%% md` / `#%% code`
uv run python tools/build_notebook.py mysource.txt notebooks/theory/01_foo.ipynb

# execute it end to end and fail loudly on any error
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 notebooks/theory/01_foo.ipynb
```

Two traps on this development machine, both of which cost me time already:

1. **This box is heavily shared** (load average ~270 on 255 cores). PyTorch
   defaults to one thread per core and thrashes. Put this at the top of any
   test script — it is a dev-box workaround and does **not** belong in the
   notebooks:
   ```python
   import os; os.environ["OMP_NUM_THREADS"] = "4"
   import torch; torch.set_num_threads(4)
   ```
2. **Run test scripts with `python -u`.** Piping to `tail` buffers stdout, and
   a script that finished in 5 seconds looks like a hang.

### Vectorize over the parameter dimension

A Python `for t in range(n_days)` loop is unavoidable for a sequential water
balance, but you should almost never loop over *parameter sets* or *basins*.
Put them on a leading batch dimension and the loop body stays vectorized:

```python
S = torch.full((n_param,), S0)          # not a scalar
for t in range(n_days):                 # loop time only
    S = S + P[t] - et(S) - q(S)
```

Measured here: 900 parameter sets x 1825 days runs in **0.2 s** this way. The
same work as 900 sequential simulations does not finish in two minutes.

# Research Brief — Differentiable & Hybrid Modeling for Hydrologic Systems

**Purpose.** This is the shared factual and notational foundation for the three notebook
authors. Treat it as the source of truth. Where a claim is uncertain, or where our copy of a
source is incomplete, that is stated explicitly rather than papered over — do not upgrade a
hedge into a confident claim when you write it into a notebook.

**Source provenance note.** `refs/shen_diff.pdf` is the *arXiv/author* version of Shen et al.
(2023). It carries the line "ADDITIONAL CLASSES, CHALLENGES TO DG, AND CONCLUDING REMARKS
REDACTED BEFORE PAPER ACCEPTANCE." Only Class I of the published taxonomy survives in this
copy, and the paper's own "Challenges" section is absent. Anything below attributed to Shen
comes from text that *is* present in our copy. Do not cite Shen for a numbered class list.
`refs/lamichhane_swe.pdf` and `refs/kalauni_lad.pdf` are both *submitted manuscripts*, not
published articles — cite them as preprints (see `references.bib`).

---

## 1. One notation for the whole workshop

The two framing papers use incompatible symbols, so somebody has to arbitrate. Here is the
scheme, then the justification.

| Symbol | Meaning | Shen et al. write | Rackauckas et al. write |
|---|---|---|---|
| `u(t)` in R^n | model **state** vector (SWE, soil store, routing store) | `u` | `u` |
| `x(t)` in R^m | dynamic **forcings** (`prcp, tmax, tmin, srad, vp`) | `x` | absorbed into `f` |
| `a` in R^k | static catchment **attributes** | `A` | — |
| `θ` in R^p | interpretable **physical parameters** (SNOW-17 `MFMAX`, HBV `β`, roughness `z0`) | `θ` | `p` / part of `θ` |
| `φ` in R^q | **neural network weights** | `W` | `θ` |
| `NN_φ(·)` | the **learnable closure**: the thing we ask the data to supply | `NN_W` | `U_θ` ("universal approximator") |
| `f(·)` | the **known physics** / RHS we keep | `g` (also `g` for the NN) | `f` |
| `y` | model outputs and diagnostics (`Q`, SWE, ET) | `y` | — |
| `y_obs` | observations | `y*` | `d` |
| `L` | loss | `L` | `C` |
| `t`, `Δt` | time, timestep | | |
| `z` | vertical coordinate / depth | | |
| `ψ` | pressure head (Richards) | | |
| `h`, `c` | LSTM hidden and cell state | | |

**Why these choices.**

- `θ` = *physical* parameters, not NN weights. Rackauckas uses `θ` for the network, but every
  hydrologist in the room already reads `θ` as "the thing SCE-UA calibrates," and both presenter
  papers use it that way (Lamichhane Table 1; Kalauni: "Physics parameters (θ) are calibrated with
  CMA-ES"). Keeping `θ` physical is the single most important decision here, because the whole
  pedagogical arc is "`θ` used to be a number you searched for; now it can be a *function*."
- `φ` = NN weights rather than Shen's `W`. `W` is needed for a single dense layer's weight matrix
  (`y = Wh + b`) the moment anyone opens up an LSTM cell, and in Rackauckas `W(t)` is the Wiener
  process. `φ` is free everywhere.
- `NN_φ` rather than `U_φ`. It is self-documenting in prose ("the NN_φ term"), reads aloud, and maps
  almost exactly onto Shen's `NN_W`. Say once, early, that Rackauckas's `U_θ` is our `NN_φ`.
- `f` for the retained physics. Shen uses `g` for the physics in the body text *and* `g` for the
  neural network in Table 1 — a genuine ambiguity in the source. Use `f` (standard ODE convention,
  matches Rackauckas) and never use `g` for either.
- `x` = forcings, `z` = space. Shen's `x` is forcings; the PINN literature's `x` is a spatial
  coordinate. Since the only spatial dimension in this workshop is depth, `z` resolves it with no
  friction and matches Kalauni's `T(z,t)` and the Richards convention. Correspondingly `ψ` (not `h`)
  is pressure head, leaving `h` free for the LSTM hidden state.

**The five canonical forms.** Every hybrid model in the workshop should be written as a
specialization of one of these, with the same symbols:

```
du/dt = f(u, x, θ, t)                              # (0) pure physics
du/dt = f(u, x, NN_φ(x, a, u), t)                  # (1) learned parameters
du/dt = f(u, x, θ, NN_φ(u, x, a), t)               # (2) learned process / closure
du/dt = f(u, x, θ, t) + NN_φ(u, x)                 # (3) learned residual (inside the ODE)
du/dt = NN_φ(u, x, t)                              # (4) neural ODE — no physics left
```

Discrete daily form, which is what the notebooks will actually run:

```
θ_t     = g_hat( NN_φ(x_{1:t}, a) )     # g_hat = sigmoid range-map onto physical bounds
u_{t+1} = u_t + Δt * f(u_t, x_t, θ_t)
y_t     = obs_operator(u_t, x_t, θ_t)
L(φ, θ) = (1/T) * Σ_t loss(y_t, y_obs_t)  ;   update  φ <- φ − η ∇_φ L
```

### 1b. Collisions you will hit in the presenter's existing material

The course notebooks and chapter 12 do **not** currently use a single scheme, so if you lift code or
LaTeX you must re-symbol it. The three real collisions:

- **`θ` is triple-booked.** In `0_neural_odes` and `1_neural_ode_adjoint` it is NN weights; in
  `2_simple_hybrid_model` it is a *concatenated* physical-plus-neural vector
  (`theta[0:4]` physical, `theta[4:9]` neural); and in `3_richards_equation_pinn_hybrid` it is
  **volumetric water content**, a physical state — while that same notebook *also* writes
  `ĥ(z, t; θ)` for the PINN's weights. Our scheme resolves this: `θ` = physical parameters, `φ` = NN
  weights, and Richards' water content should be written **`θ_w`** (or `ϑ`) with the collision called
  out once in prose, since `θ` for water content is too entrenched in soil physics to rename silently.
- **`h` is double-booked**: LSTM hidden state in the RNN notebooks, pressure head in Richards. Our
  scheme: `h` = hidden state, `ψ` = pressure head.
- **`T` is double-booked**: terminal time in `1_neural_ode_adjoint`, sequence horizon in `0_rnn_theory`,
  and **temperature** in `2_simple_hybrid_model`. Our scheme: `T` = temperature (it is a forcing in
  `x`), `t_end` for terminal time, `L` for lookback/sequence length.

Also note `chapter_12` uses `k` for reservoir conductivity (negative, `dS/dt = kS`) while
`2_simple_hybrid_model` uses `k` for a positive recession coefficient, and `c` means different
exponents in each. Pick one bucket model and do not mix their symbols.

## 2. Definitions, kept apart

These terms get conflated constantly. They are not synonyms.

- **Differentiable modeling** (Shen's "Differentiable Geosciences", DG) is a *property of the
  implementation*: the entire forward workflow, physics included, supports accurate and cheap
  gradient computation end-to-end. Shen is explicit that this includes non-AD routes: "we use the
  term differentiable modeling to include any method that can produce the gradients rapidly and
  accurately at scale," naming hand-derived adjoint methods as a non-AD example. Differentiability
  is a *capability*, not an architecture. It is normally only needed for training, not for forward
  runs.

- **UDE (universal differential equation)** is a *mathematical object*: "differential equations
  which are defined in full or part by a universal approximator." Rackauckas's general form is
  `N[u(t), u(α(t)), W(t), U_θ(u, β(t))] = 0` — a forced stochastic delay PDE with embedded
  approximators; every case in this workshop is the trivially reduced ODE version. "Universal"
  refers to universal approximation (Hornik 1991), not to generality of the equation.

- **Neural ODE** is the *special case of a UDE defined entirely by the network*:
  `du/dt = NN_φ(u, t)`. Rackauckas says this explicitly. It is form (4) above. Chen et al. (2018) is
  the origin; its main technical contribution for our purposes is the continuous adjoint, which buys
  O(1) memory — see §6, pitfall 5.

- **PINN** is a *different problem setup entirely*, and the workshop must not blur this. A PINN
  trains a network to *be the solution surface*: `ψ_hat = NN_φ(z, t)`, with a loss that penalizes
  (i) mismatch with observed data points and (ii) the PDE residual, evaluated by differentiating the
  network with respect to its own *inputs*. Shen's framing: the network's inputs are space-time
  coordinates, "(i) h(t,x) agrees with known data points at (t,x), and (ii) the derivatives dh/dx,
  dh/dt, etc. agree with the governing partial differential equations." Consequences to state
  plainly: a PINN is tied to one initial/boundary-condition pair and must be retrained for another;
  it uses no classical solver; and the network replaces *the solution*, not *a process*.
  Rackauckas's criticism is precise — "PINNs frame the solution process as a large optimization …
  [it] does not incorporate the numerical techniques which have led to stable and efficient solvers."

- **"Physics-informed / physics-guided ML" (PGML)** is the broad, loose family. Shen draws the
  distinction sharply in Supplementary Discussion C: PGML adds physics as *regularization or
  pre-training* to an ML model, "does not in theory need differentiable programming," treats the
  physical law *as truth*, and aims to make the ML model more robust. DG demands full end-to-end
  differentiability, uses the numerical model as the backbone, and — the philosophical difference —
  "we do not presume the physical laws to be correct, and, rather, are constantly looking for
  opportunities to update existing knowledge."

**Nesting, stated once:** every neural ODE is a UDE; every UDE implementation is a differentiable
model; not every differentiable model is a UDE (a differentiable solver used only for calibration
has no embedded network); PINNs overlap differentiable modeling but are a distinct formulation;
PGML is a superset that includes things which are neither end-to-end differentiable nor hybrid.

## 3. Taxonomy of hybrid designs

Shen's body text names three places to put the question mark (their items i–iii, quoted below). We
add two more that the literature actually uses, and flag one that Shen deliberately excludes.

| # | Design | Form | Hydrologic example |
|---|---|---|---|
| A | **Learned (static) parameters** — "differentiable parameter learning" (dPL) | `θ = NN_φ(a)`, then `du/dt = f(u,x,θ)` | Tsai et al. (2021): an NN maps catchment attributes to VIC parameters, trained *through* VIC on all sites at once. Shen's item (i): `y = g(u, x, θ = NN(A))`. |
| B | **Learned time-varying parameters** — dynamic parameterization | `θ_t = g_hat(NN_φ(x_{1:t}, a))` | **Lamichhane & Bennett**: an LSTM emits eight SNOW-17 parameters *at every timestep*, sigmoid-mapped onto the published parameter ranges. |
| C | **Learned process / closure** — an internal flux or constitutive relation | `du/dt = f(u,x,θ, NN_φ(u,x,a))` | **Kalauni et al.** "Resistance NN": replaces aerodynamic `ra` and stomatal `rs` but keeps the bulk-transfer equations for `H` and `LE`. Shen's items (ii)–(iii); Rackauckas's Boussinesq closure `wT = U_θ(P, T, ∂T/∂z)`. Also `3_richards`: learn `θ_w(ψ), K(ψ), C(ψ)` and keep the Celia solver. |
| D | **Learned module wholesale** — an entire governing equation replaced | `H, LE = NN_φ(u, x, a)`, ODE structure retained | **Kalauni et al.** "Full Flux NN". Still hybrid: the ODE system still advances soil moisture and temperature, and the network still sees model *states*. |
| E | **Differentiable solver only** — no network at all | `du/dt = f(u,x,θ)`, grad of L wrt θ by AD | Gradient-based calibration of a conceptual model. Best short demo of "why gradients": same model, same data, no ML, orders of magnitude fewer forward runs than SCE-UA. |
| — | *Learned residual as a post-processor* | `y = f(...) + NN_φ(x)` fit to PBM errors, trained **offline** | Shen puts this **outside** DG: "training ML models to predict the PBM residuals" appears among "not-fully-differentiable methods … outside of the scope of DG." Mention it as what people usually mean by "hybrid," and say why it is weaker: the network cannot see the model's internal states and cannot adapt when the physics changes. |

A residual term placed *inside* the ODE and trained *through* the solver (form 3 in §1) **is** a
UDE — Rackauckas's Lotka–Volterra example is exactly that, and so is `2_simple_hybrid_model`'s
`r_θ(S, P, T)`. The distinction is online vs. offline training, not the word "residual."

Shen's Class I, the taxonomy statement that does survive in our copy, is "Directly differentiating
through numerical models and connecting them to NNs" — reimplement the existing model on a
differentiable platform, then attach networks. Costs Shen names honestly: "reimplementing a model
does incur non-trivial initial development cost," and "mathematical changes may be required to adapt
previously non-differentiable mathematical operations."

## 4. Why differentiability matters — the "why should I care"

This is the argument to make to a room of people who already own a working calibration workflow.

**Parameter count is the whole game.** Shen: LSTMs "widely employed in hydrology can contain
~500,000 weights," while "traditional evolutionary, or genetic or particle swarm optimization
methods can hardly handle more than a few dozen independent parameters." Derivative-free search
scales badly in dimension; gradient descent does not care. Concretely, using the workshop's own
numbers (verified by running PyTorch 2.14 in this repo's venv):

- SNOW-17 as published: **10 parameters, 8 calibrated**. Comfortably inside SCE-UA's reach.
- A 1-layer LSTM, 7 inputs, 64 hidden units, linear head to 8 parameters: **19,208 weights**.
- Kalauni's MLP (7 -> 128 -> 128 -> 128 -> 2, tanh): **34,306 weights**.

No derivative-free algorithm reaches 19,208. Reverse-mode AD returns the full gradient for roughly
the cost of one extra forward pass. Shen's counterpoint on finite differences: "10,000 weights would
require 10,001 forward model evaluations."

**Wall clock, from the literature.** Tsai et al. (2021), as described in Shen's Supplement:
differentiable parameter learning replaced a job that "normally takes a 100-CPU cluster 2-3 days"
with "a single GPU one hour."

**Regionalization and PUB — the deeper reason.** Conventional calibration fits `θ` *per site*, then
needs a separate regionalization step to reach ungauged basins, and that step is where the
uncertainty lives. A differentiable model learns the mapping `a -> θ` (or `x, a -> θ_t`) **jointly
across all sites under one global loss**, so regionalization stops being a post-hoc regression and
becomes the thing being trained. Consequences reported in the sources:

- Shen/Tsai: dPL parameter fields are "spatially coherent," "extrapolate better in space," and the
  approach "address[es] the notorious problem of parameter equifinality."
- Feng et al. (2022), as summarized in Shen: delta-HBV reached **median NSE 0.732 vs. LSTM's 0.748**
  on the CAMELS streamflow benchmark (0.715 vs. 0.722 on a second forcing dataset) — essentially
  LSTM-level accuracy from a model that also emits ET and baseflow. In spatial-extrapolation tests
  the differentiable model **outperformed** the LSTM.
- **Kalauni et al.**: a single regional network conditioned only on static attributes matched or beat
  physics *that had been calibrated individually at each held-out site* — "at least 75% of sites for
  sensible heat and 65% for latent heat."
- **Lamichhane & Bennett**: one regional hybrid model replaced 734 site-wise calibrations and did
  better.

**The other three payoffs, ordered by how much they land with this audience:** (1) you still get
every internal flux and state, so the model can answer questions about unobserved variables and give
a narrative; (2) you can place the question mark precisely — Shen's framing is that traditional
inversion only ever asks "θ = ?" while differentiable models let you ask "*g* = ?"; (3) the same
gradients are reusable for sensitivity analysis, data assimilation, and trajectory optimization.

## 5. Key results to cite from the presenter's two papers

### Lamichhane & Bennett — SNOW-17 + LSTM dynamic parameterization (submitted to WRR)

*Setup.* SNOW-17 reimplemented in PyTorch. A 1-layer LSTM (64 hidden units, sequence length 365 d,
dropout 0, Adam, lr 1e-3, batch 256, MSE loss) consumes daily precipitation and min/mean/max air
temperature plus latitude and elevation, and emits eight of SNOW-17's ten parameters **at every
timestep**, squashed by a sigmoid onto the published ranges (Table 1: `SCF` 0.7–1.6, `MFMAX`
0.6–2.5, `MFMIN` 0.05–0.59, `PXTEMP1` -3–0 degC, `PXTEMP2` 0–3 degC, `UADJ` 0.05–0.20, `TIPM`
0.1–1.0, `PLWHC` 0.005–0.40; `NMF` and `MBASE` held fixed following He et al. 2011 sensitivity).
Trained end-to-end on **734 SNOTEL sites** across the CONUS. Baselines: SNOW-17 calibrated per site
with **SCE-UA** on RMSE, and a same-architecture LSTM predicting SWE directly. Three experiments:
(1) site-wise, (2) regional with a 70/15/15 chronological split, (3) spatial holdout — four clusters,
4-fold spatial cross-validation.

*Headline numbers (median KGE across test sites).*

| Experiment | SNOW-17 | Hybrid | LSTM |
|---|---|---|---|
| 1 — site-wise | 0.740 | 0.783 | 0.826 |
| 2 — regional, temporal split | (0.740; SNOW-17 is site-wise only) | ~0.863 | ~0.863 |
| 3 — spatial holdout (PUB) | (0.740) | 0.824 | 0.843 |

*The result that actually matters, and the one to build the workshop's punchline around:* KGE cannot
tell the hybrid and the LSTM apart, but **melt-out duration** can. Median melt-out-duration error,
Experiment 2: hybrid **+1.0 days**, LSTM **+11.75 days**. Experiment 3: hybrid **+0.93 days**, LSTM
**+15.72 days**. The LSTM retains a trace snowpack late into the season; SNOW-17's structure forbids
it. The finding is insensitive to the 5 mm melt-out threshold (checked in the supplement). Quote
worth using verbatim: "traditional metrics may not fully capture physically meaningful improvements."

*Other numbers.* Hybrid median bias -3.7 mm vs. LSTM +1.9 mm (Exp. 2); bias spread at unseen sites
27.92 mm (hybrid) vs. 42.62 mm (LSTM). The hybrid *under*-predicts peak SWE (-19 mm median, Exp. 2)
— an inherited SNOW-17 tendency, and an honest cost of the physics. Ten sites had KGE < 0, all in
the Cascades or southern AZ/NM: mean SWE 26.6 mm vs. 244.7 mm at good sites, winter mean temperature
+0.48 degC vs. -1.32 degC. Warm, ephemeral snow breaks both models.

### Kalauni, Gupta & Bennett — hybrid models as diagnostic tools (submitted to WRR)

*Setup.* The Land-atmosphere Dynamics (LaD) model (Milly & Shmakin 2002) reimplemented in JAX +
Equinox + Diffrax; four-component state `(w_s, w_r, w_g, T(z,t))` with a 5-layer soil-heat column
(0.05, 0.10, 0.35, 1.0, 3.0 m); **Kvaerno5** implicit ESDIRK solver (stiff), rtol 1e-3, atol 1e-4.
**135 FLUXNET2015 sites**, half-hourly, 43 static attributes (6 numeric + 37 one-hot). Physics
parameters calibrated per site with **CMA-ES**. Three configurations spanning inductive bias:
Physics Only -> Resistance NN (learn `ra, rs`, keep the bulk-transfer equations) -> Full Flux NN
(learn `H, LE` directly). MLP 3x128 tanh in both hybrids.

*Headline numbers.* Per-site training, median over 135 sites:

| Flux | Metric | Physics | Resistance NN | Full Flux NN |
|---|---|---|---|---|
| H | NSE | 0.61 | 0.72 | **0.77** |
| H | RMSE (W/m2) | 51.9 | 42.0 | **37.1** |
| LE | NSE | 0.59 | 0.68 | **0.70** |
| LE | KGEss | 0.72 | **0.78** | 0.73 |

Regional / PUB (held-out sites, compared against physics calibrated *at those very sites*): H NSE
0.61 -> **0.73** for both networks; LE NSE 0.59 -> **0.65** for both. The two networks are
statistically indistinguishable on every variance-based metric (all p > 0.2); the only reliable
difference is bias, and there the *more constrained* Resistance NN wins (median absolute bias smaller
by 2.2–2.5 W/m2). **Hence the paper's thesis: physical inductive bias is an asset for regional
generalization, not a handicap.**

*The diagnostic results — this is the "hybrid as instrument" argument.* At unseen sites, observed LE
peaks at **107 W/m2** near local noon; calibrated physics gives **91 W/m2**, roughly an hour late;
both hybrids give **105 and 107 W/m2**, in phase. Binned by vapour pressure deficit, observed LE
rises then *declines*; the physics model plateaus (its stress function depends on soil moisture
alone); both networks recover the decline. Because the Resistance NN keeps the bulk-transfer
structure, the improvement is attributable *purely to the resistances* — a specific, actionable
finding: stomatal resistance needs an explicit VPD dependence, not just a soil-moisture dependence.

*The honest counter-result, which the workshop must not omit.* On the long-term evaporative index
`E/P` at unseen sites, per-site-calibrated physics wins: Spearman rho **0.88** (physics) vs. **0.68**
(Resistance NN) vs. **0.66** (Full Flux NN). The hybrids nail the sub-daily dynamics and lose the
climatology. Partly a training-objective artefact — both were trained on variance-based losses that
do not penalize bias — but the physics models were calibrated to a variance-based objective too, so
it is not purely that.

## 6. Pitfalls and gotchas

1. **Gradients through long unrolled simulations.** A 30-year daily run is a 10,957-step recurrence;
   gradients vanish or explode exactly as in an RNN. Kalauni: vanishing/exploding gradients are
   "exacerbated at half-hourly resolution." Mitigations actually used in these papers: train on short
   windows and *carry the state forward* (Kalauni: 480 half-hourly steps ~ 10 days, stride 460, with
   the ODE state at the stride index seeded into the next window — this preserves hydrologic memory
   without an intractable graph); gradient clipping at norm 1.0; truncated BPTT; a spin-up period
   excluded from the loss. The presenter's own `3_richards` notebook makes the same call in a code
   comment: a 240-node-deep graph means "gradients vanish before reaching the network weights,"
   which is why it unrolls only 8 steps with 3 Picard iterations.
2. **Stiffness.** Kalauni chose an implicit ESDIRK solver because the soil heat-diffusion PDE is
   stiff, and reports that "accurate integration requires double-precision arithmetic … substantially
   slower and more memory-intensive." Rackauckas: reverse-solve adjoints are "known to be unstable …
   such as on stiff equations," and for stiff/DAE systems `BacksolveAdjoint` is "almost certainly
   unstable." Related open risk (Kalauni section 4.3): the network may learn to **compensate for
   solver error** rather than improve the physics, which would defeat the whole point.
3. **Non-differentiable operations.** Thresholds, `if`, `min`, `max`, hard clamps. Shen notes these
   are piecewise differentiable so AD *will* return something — which is worse than an error, because
   the gradient is zero wherever you sit on a flat branch. Kalauni replaces "all discontinuous
   operations (e.g., threshold-based snow/rain partitioning, non-negativity enforcement) … with
   smooth sigmoid-based approximations with steepness parameter k = 20," and uses a soft upper clamp
   `r = r_max - softplus(r_max - r_tilde)` explicitly "because it retains a non-zero gradient when
   the output approaches the ceiling, which preserves gradient flow during training." Standard
   replacements to teach: hard rain/snow threshold -> `sigmoid(k*(T_thresh - T))`; `min(a,b)` ->
   smooth-min; `relu` -> `softplus`; `clamp(x, 0, x_max)` -> nested softplus. Nice teaching moment:
   SNOW-17's dual `PXTEMP1`/`PXTEMP2` partitioning is *already* a linear ramp — hydrologists have
   been writing smooth relaxations for decades without calling them that.
   PyTorch-specific: `torch.where` is differentiable but will propagate `NaN` gradients from the
   *untaken* branch if that branch computes a `NaN` (`sqrt` of a negative, divide by zero) — mask the
   **input**, not the output. And in-place writes into a tensor that needs grad (`u[t] = ...`) will
   either raise a version-counter error or break the graph; accumulate into a Python list and
   `torch.stack`. The presenter's `thomas_solve_torch` in `3_richards` carries a docstring saying
   exactly this.
4. **Initialization and scaling.** An untrained network inside a physics model emits nonsense
   parameters and the model diverges before the first useful gradient. Fixes used in the sources:
   squash NN outputs onto published physical ranges (Lamichhane: sigmoid onto Table 1 ranges;
   Kalauni: softplus floor + soft ceiling giving `ra` in [1, 2000] and `rs` in [10, 10000] s/m, with
   scale and shift constants fixed from training-data statistics and *not learned*; chapter 12's
   `HydroParam(low, high, net)` class does the same thing with a `register_buffer`-backed sigmoid
   rescale, which is the cleanest reusable implementation in the presenter's own code);
   **supervised pre-training** of the NN against the physics formula's own output before going online
   (Kalauni: 5000 steps, Adam 3e-4, batch 4096, log-space MSE for the wide dynamic range of `ra`);
   standardize inputs using *training* statistics only. Rackauckas's own recipe is a two-stage
   optimizer: 200 Adam steps at lr 0.1, then BFGS, converging in 400–600 total iterations. General
   rule worth stating out loud: **initialize so that the hybrid at step 0 reproduces the calibrated
   physics model**, then let training improve on it.
5. **Adjoint vs. discretize-then-optimize.** *Discretize-then-optimize* (DTO) = run the solver, tape
   every operation, backprop through the tape. Exact gradients of what you actually computed; memory
   O(number of steps). *Optimize-then-discretize* (the continuous adjoint, Chen et al. 2018) = derive
   the adjoint ODE and solve it backwards; O(1) memory, but the gradient is of the continuous problem
   and reverse integration can be unstable. Rackauckas is blunt: the reverse-solve adjoint is "the
   common adjoint utilized in neural ODE software such as torchdiffeq," and "torchdiffeq's adjoint
   calculation diverges on all but the first two examples" in their benchmark suite. Rackauckas
   catalogues eight adjoint modes with a decision tree; the two rules that transfer to PyTorch are
   *use forward-mode below roughly 50 parameters+states* and *never use the reverse-solve adjoint on
   a stiff system*. **Practical rule for this workshop:** a daily, fixed-step, explicit hydrologic
   model should just be a Python `for` loop with autograd taping it — that is DTO, it is exact, and
   10,000 daily steps of a small state vector fits in memory. Use `torchdiffeq.odeint` (DTO) for the
   neural-ODE demo; reach for `odeint_adjoint` only to *show* the memory/stability trade-off, not as
   the default. The presenter already has a hand-written `torch.autograd.Function` adjoint and a
   gradient check against direct autograd in `2_simple_hybrid_model` — reuse that, it is the cleanest
   way to make the distinction concrete in five minutes.
6. **Splitting in time, and in space.** Both presenter papers split chronologically, never randomly:
   Lamichhane 70/15/15; Kalauni 70/30 plus stratified 5-fold spatial CV and a PUB holdout. Normalize
   with training-period statistics only. The strong test is spatial holdout; the temporal split is the
   easy one. Also worth stating: Kratzert et al. (2024), "never train an LSTM on a single basin" —
   Lamichhane's Experiment 1 exists only as a benchmark and the authors say so in the text. The
   presenter's `1_lstm_leaf_river` states the rule crisply: "For time series we **never shuffle**:
   past cannot be used to predict past."
7. **Equifinality does not disappear.** Beven (2006) still applies. Differentiable parameter learning
   *reduces* it by pooling a global loss across sites, but does not abolish it, and a time-varying
   `θ_t` reintroduces enormous freedom — Lamichhane's own caveat is that the hybrid "does not impose
   strict mass-balance constraints" because rain/snow partitioning and melt factors are learned.
8. **Metrics hide the physics.** Both presenter papers make this point independently: KGE cannot
   separate hybrid from LSTM in Lamichhane while melt-out duration separates them by two weeks; NSE
   and KGEss rank Kalauni's two configurations in opposite orders. Always pair an aggregate score
   with a process-specific diagnostic.
9. **Offline-trained networks can be stable offline and unstable when coupled.** Kalauni cites
   Brenowitz et al. (2020) and Wang et al. (2022) for exactly this in climate models. Online training
   through the solver mitigates it; it does not eliminate it.

## 7. Open questions / research frontier (for Agent 4)

- **From learned function to written equation.** Rackauckas's UDE + sparse symbolic regression
  recovers Lotka–Volterra's missing quadratic terms where SINDy on splined derivatives fails — but
  the honest number from their own robustness study is a **(50.4 +/- 25.7)% recovery rate** across
  498 error-free runs. Kalauni: "How to distill them into interpretable and sensible equations, on
  the other hand, is an open question that we take up in an upcoming study." This is the single most
  important frontier for the "so what did we actually learn?" discussion.
- **Does the network fix the physics or the solver?** Kalauni section 4.3 raises it and does not
  resolve it.
- **Long-term balances out of sample.** Kalauni: hybrids win the diurnal cycle and lose the
  climatological `E/P` at unseen sites. Whether bias-aware or multi-objective losses fix this is open.
- **Is the inherited state vector the right one?** Kalauni: "There is no guarantee that the state
  vector we inherit from the LaD model is either sufficient or parsimonious," pointing toward learned
  latent coordinates (Champion et al. 2019).
- **Uncertainty.** Shen's abstract closes on it: "Future work should address computational challenges,
  reduce uncertainty, and verify the physical significance of outputs."
- **Nonstationarity and climate-change projection.** Shen argues stronger priors should help; the
  cited evidence is preliminary (Feng et al. 2023).
- **How much physics is the right amount?** Kalauni's whole design — a spectrum of inductive bias —
  is the cleanest available framing of this question, and it has no general answer yet.
- **Shen's wish list**, useful verbatim as a discussion prompt: accuracy matching pure ML; models
  capable of *structural evolution*; generalization to data-sparse regions and the long-term future;
  conservation of mass/energy/momentum; internally consistent fluxes that support a full narrative;
  and the ability to "isolate one uncertain model component at a time to learn physics with less
  ambiguity."
- **Shen's seven question types**, also good discussion fuel: (a) relationship between two variables,
  (b) missing physics in a differential equation, (c) what the assumption/function should have been,
  (d) how a factor influences a parameter, (e) which process causes a phenomenon, (f) behaviour under
  new environmental conditions, (g) information content of a dataset.

## 8. Hard constraints for the notebooks

- **Package.** `pip install git+https://github.com/BennettHydroLab/minicamels.git`;
  `from minicamels import MiniCamels`. Already installed and verified in this repo's `.venv`
  (v0.1.dev7, alongside `torch` 2.14.0 and `torchdiffeq`, all importing and running). **Undeclared
  dependency:** `minicamels` needs `requests` and `aiohttp` for its remote loading path but does not
  declare them; the project `pyproject.toml` already pins both, so do not strip them.
- **Verified API** (checked live against the installed package, not recalled):
  `ds = MiniCamels()` -> `ds.basins()` (DataFrame, 50 rows, columns `basin_id`, `basin_name`),
  `ds.attributes()` (DataFrame indexed by `basin_id`), `ds.load_basin(id)` / `ds.open_basin(id)`
  (xarray Dataset), `ds.open_basins()`, `ds.load_all()`, `ds.get_forcings(id, start, end)`,
  `ds.get_streamflow(id, start, end)`, `ds.get_water_year(id, wy)`, `ds.plot_basin()`, `ds.plot_map()`.
- **Data shape.** 10,957 daily steps per basin, 1980-10-01 to 2010-09-30 (WY1981–WY2010, so
  water-year alignment and spin-up are natural). Variables `prcp` (mm/d), `tmax`, `tmin` (degC),
  `srad` (W/m2), `vp` (Pa), `qobs` (mm/d). **No NaNs** in the basin inspected. Data are fetched from
  GitHub raw URLs on first use — a workshop room on flaky wifi should pre-cache, and `load_all()`
  issues 50 HTTP requests.
- **Correction to the planning assumption:** `attributes()` returns **16** columns, not ~18. They are:
  `lat, lon, elev_mean, slope_mean, area_km2, mean_prcp, mean_pet, aridity, frac_snow, q_mean,
  runoff_ratio, hfd_mean, baseflow_index, soil_depth_pelletier, frac_forest, lai_max`.
  (The minicamels README says ~18; the installed package returns 16. Trust the package.)
  Note also that `mean_pet` is a **static scalar attribute**, not a timeseries — there is **no PET
  forcing variable**. A notebook needing daily PET must compute one (Hamon, or Priestley-Taylor from
  `srad`/`tmax`/`tmin`/`vp`) and say so in prose.
  Note `frac_snow` and `aridity` — directly useful both for basin selection and as NN inputs. Across
  the 50 basins `frac_snow` spans 0.000–0.777 (median 0.113) and `aridity` spans 0.23–5.21;
  `elev_mean` 22–2256 m; `area_km2` 50–5319; `mean_prcp` 0.64–8.15 mm/d.
- **There is NO observed SWE in minicamels, and no observed ET.** This is the binding design
  constraint on any SNOW-17-flavoured example. Three workable routes, in order of preference:
  1. **Synthetic-truth twin experiment.** Run a "true" snow or bucket model with known `θ` (or a
     known closure) on real minicamels forcings, treat its output as pseudo-observations, and train
     the hybrid to recover it. You then get an exact answer to plot the learned function against —
     the single most convincing figure in this whole subject, and precisely what Rackauckas does with
     Lotka–Volterra and what the presenter already does twice (chapter 12's `plot_conductivity`,
     `3_richards` cell 16). Strongly recommended for the "did we actually learn the missing physics?"
     moment.
  2. **Train the snow component on streamflow.** Defensible and citable: Shen notes "streamflow can
     constrain a model to better simulate snow water equivalent," citing Jiang et al. (2020). Be
     explicit that SWE is then only weakly identified — that honesty is itself a teaching point, and
     it sets up the equifinality discussion.
  3. **Retarget to streamflow entirely** (a conceptual bucket / HBV-lite), where `qobs` is a real
     observed target and the hybrid story needs no apology.
  Do **not** silently present a simulated SWE time series as validated. If a notebook plots SWE
  against nothing, say so on the figure.
- SNOW-17 as published uses **6-hourly** melt factors (`mm degC^-1 per 6h`); minicamels is daily. Any
  SNOW-17 adaptation must state the conversion rather than hide it.
- **Framework: PyTorch** throughout (matches the chapter and the course). `torch` 2.14 and
  `torchdiffeq` 0.2.5 are installed. Kalauni's paper is JAX/Diffrax — mention the ecosystem, do not
  switch frameworks mid-workshop.
- **Budget: 120 minutes total, CPU laptop.** No live training run should exceed ~60–90 seconds;
  anything longer must be precomputed, shrunk, or shipped as cached weights. See
  `resources/reusable_assets.md` for which existing cells are already over budget and by how much.

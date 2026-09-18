# Reusable Assets Inventory

*What existing code and explanation to adapt, where it lives, and what it costs.*

Audience: Agents 2, 3 and 4. All paths absolute. Cell indices are 0-based positions in the
notebook's `cells` array — verify with
`python -c "import json,sys; nb=json.load(open(sys.argv[1])); [print(i,c['cell_type'],len(''.join(c['source']).split(chr(10)))) for i,c in enumerate(nb['cells'])]" <path>`
before copying, since editing a notebook shifts every index after the edit.

Read `research_brief.md` §1 and §1b first. **Nothing below complies with the workshop notation as
written** — every asset needs re-symboling, and the required renames are listed per asset.

---

## 0. Read this before you copy anything

**Five facts that change what is copyable.**

1. **No MyST admonitions exist anywhere in the source material.** Not one ` ```{note} `, `:::{...}`,
   `{warning}` or `{admonition}` across all seven notebooks. The only callout device in the corpus is
   the plain Markdown blockquote, used in chapter 12 as `> ### A note on <topic>` (4 instances) and
   once in `1_neural_ode_adjoint`. If the workshop uses MyST admonitions that is a **new** convention
   — decide it once, centrally, and apply it uniformly. The natural mapping is chapter 12's
   `> ### A note on X` → ` ```{note} `.
2. **`chapter_12` cannot run as shipped.** Cell 67 does
   `xr.open_dataset('./data/camels_data.nc')`, which resolves to
   `/home/andrbenn/workspace/differentiable_modeling_workshop/refs/data/camels_data.nc` — **the file
   does not exist**. `assets/computational_graph.png`, embedded by cell 5's markdown, is also missing.
   The chapter halts at cell 67. Everything before that point runs; everything after must be rebuilt
   on `minicamels`.
3. **`minicamels` is not used by any source notebook.** Chapter 12 predates it and reads a raw
   NetCDF. Every data-loading cell in the corpus must be rewritten.
4. **Two different Python eras.** The five `computational_methods_course` notebooks are Python 3.13.5
   and run clean. `chapter_12` is Python 3.8.12 with torch 1.10.1 / xarray 2022.3.0 pinned in its own
   prose, and uses xarray APIs (`Dataset.drop`, `.drop(['quantile','season'])`) that are removed or
   deprecated in current xarray. **Do not copy chapter 12's xarray code.** Its PyTorch code is fine.
5. **There are no checkpoints or saved weights anywhere** — no `torch.save`, no `.pt` files. If the
   workshop wants to ship pretrained weights (and it should, see §4), that infrastructure has to be
   built from scratch.

**Three helpers exist in mutually incompatible versions. Unify before anyone writes a notebook.**

| Helper | Versions | The trap |
|---|---|---|
| NSE | `nse(obs, pred)` (2_hybrid c15), `calc_metrics(true, pred)` (rnns/1 c15), `nse(sim, obs)` (ch12 c81) | **Chapter 12's argument order is reversed.** Silent wrong answers. |
| RK4 | `rk4_step`/`odeint_fixed` (0_neural_odes c6); a method on `RK4HybridModel` (2_hybrid c9); a free `rk4_step_raw_theta` (2_hybrid c12) | Three implementations of one algorithm. |
| Newton | `newton_solve` (ch12 c19); a method on `NeuralReservoir` (ch12 c34, with an extra `err` guard) | Two divergent copies in one document. |

**Recommendation:** put one of each in `workshop_utils/` (the package already exists in
`pyproject.toml`), settle on `nse(obs, sim)`, and import everywhere.

**Device handling is inconsistent:** `0_neural_odes` and `2_simple_hybrid_model` hard-pin
`torch.device('cpu')`; `3_richards`, `1_lstm_leaf_river` and `chapter_12` use
`cuda if available else cpu`. For a CPU-laptop workshop, **hard-pin CPU everywhere** — a participant
with a GPU should not get different timings from the person next to them.

**Strip trailing empty markdown cells on copy:** `0_neural_odes` (20, 21), `1_neural_ode_adjoint`
(18, 19), `3_richards` (20), `1_lstm_leaf_river` (24).

---

## 1. Runtime budget — the whole-notebook picture

Total workshop budget is **120 minutes including setup and discussion**. Estimates are for a CPU
laptop.

| Source notebook | Est. full runtime | Verdict |
|---|---|---|
| `1_neural_ode_adjoint` | **< 5 s** (no training at all) | **Copy freely.** |
| `0_neural_odes` | **~15–40 s** | **Copy freely**, trim epochs if you want headroom. |
| `rnns/0_rnn_theory` | ~1–2 min | Copy the explanation; shrink or drop the 3-model comparison. |
| `2_simple_hybrid_model` | ~3–6 min | **Copy the structure, cut the experiment to one method.** |
| `chapter_12` | ~5–20 min *if the missing data existed* | Copy the prose and the small modules; rebuild the data half. |
| `rnns/1_lstm_leaf_river` | ~10–40 min | **Over budget.** Cut epochs; drop the lookback sweep. |
| `3_richards_equation_pinn_hybrid` | **~20–60+ min** | **Far over budget.** See §4. |

**The runtime dials, in priority order.** These are the first things to grab:

| Dial | Location | Current | Suggested for a 2h workshop |
|---|---|---|---|
| `N_EPOCHS` (hybrid Richards) | `3_richards` cell 15 | **2000** | 200–400, or ship cached weights |
| `N_EPOCHS_PINN` | `3_richards` cell 11 | **5000** | 500–1000, or ship cached weights |
| `n_epochs` (LSTM/GRU/MLP) | `1_lstm_leaf_river` cell 12 | **60** | 10–15 |
| `lookback_values` (7 retrainings) | `1_lstm_leaf_river` cell 20 | 7 values | **drop the cell, or precompute** |
| `max_epochs` / `max_sub_epochs` | `chapter_12` cells 42, 77 | 300 / 6×2 | 50–100 / 2×1 |
| `N_TOTAL` | `2_simple_hybrid_model` cell 3 | **720** | keep at 720 — **do not let anyone set it to `len(df_full)`; that is a 15× multiplier and turns cell 16 into 30–75 minutes** |
| `n_iter` (Newton per step) | `2_simple_hybrid_model` cell 9 | **30** | 5–10 |
| `epochs` | `0_neural_odes` cell 12 | 1200 | fine as is |

---

## 2. `0_neural_odes.ipynb` — the best-value notebook in the corpus

`/home/andrbenn/workspace/differentiable_modeling_workshop/refs/computational_methods_course/notebooks/hybrid_modeling/0_neural_odes.ipynb`
22 cells (14 md / 8 code), 391 source lines, runs in ~15–40 s.

| Cell(s) | ~Lines | What it is | Recommendation |
|---|---|---|---|
| **6** | **15** | `rk4_step(func, z, t0, t1)` + `odeint_fixed(func, z0, t_eval)` | **Reuse near-verbatim — this is the single most reusable artifact in the corpus.** Promote to `workshop_utils`. Three other copies of RK4 in the corpus should be deleted in its favour. |
| 0, 2, 4 | 26+23+9 md | ResNet → continuous depth; "a ResNet is forward Euler"; `ODESolve` as a black box | **Reuse near-verbatim.** This is the cleanest statement of the core idea anywhere in the corpus. |
| 3 | 26 | Spiral demo: `solve_ivp` dense vs hand Euler at `dt=0.75` | **Reuse.** The visual that makes "a residual block is a discrete step" land in ten seconds. |
| 8 | 27 | Builds 14 irregular observations + 2-panel plot | **Compress.** Good motivation for irregular sampling; the hydrologic version is "you have 12 snow-course measurements a year." |
| 10 | 25 | `class ODEFunc(nn.Module)` (`state_dim=2, hidden_dim=32`), `class NeuralODE(nn.Module)` | **Reuse, re-symbol.** The minimal neural ODE in 25 lines. |
| 12 | 18 | Training loop, `epochs=1200`, `lr=5e-3`, `clip_grad_norm_(1.0)` | **Reuse.** ~10–30 s. Keep the gradient clipping and point at it — it is pitfall 1 in the brief. |
| 13 | 35 | 3-panel result figure (loss / phase space / interpolation) | **Reuse**, promote to `workshop_utils`. |
| 15 | 34 | True vs learned quiver fields | **Reuse but vectorize.** Currently a nested Python loop over 289 points under `no_grad` (~1 s). One `reshape` fixes it. Conceptually valuable: "what did the network learn?" as a *field*, not a curve. |
| 16 | 13 md | 5-row honesty table: what this notebook simplifies relative to Chen et al. | **Reuse near-verbatim.** Rare and valuable. |
| **17** | 33 md | Continuous normalizing flows — the NODE → CNF → diffusion ladder | **Skip.** Excellent writing, wrong workshop. Cite it as further reading. |
| 18, 19 | 23+15 md | Why neural ODEs matter for Earth-science hybrids; summary table | **Compress** — several of its four examples are better served by the presenter's own papers. |

**Re-symboling required.** This notebook uses `h(t)` for state in the depth framing and `x(t)`/`z` in
the data framing; `θ` for NN weights; `A` for the true linear operator; `f_θ` for the vector field;
and — in cell 18 — `φ` for *physical* parameters and `θ` for *neural* ones, which is exactly backwards
from the workshop scheme. Rename throughout: state `h`/`x`/`z` → **`u`**; NN weights `θ` → **`φ`**;
physical parameters `φ` → **`θ`**; `f_θ` → **`NN_φ`** when the whole RHS is learned. Keep `f` for the
known physics.

Cell 9's hybrid split `dx/dt = g_phys(x,t) + g_nn(x,t)` is the corpus's canonical statement of the
learned-residual form and should be **kept but rewritten** as `du/dt = f(u, x, θ, t) + NN_φ(u, x)` —
i.e. form (3) of the brief. Do not keep `g` for either half; the brief forbids it because Shen uses
`g` for both.

---

## 3. `1_neural_ode_adjoint.ipynb` — free, and it answers the question everyone asks

`/home/andrbenn/workspace/differentiable_modeling_workshop/refs/computational_methods_course/notebooks/hybrid_modeling/1_neural_ode_adjoint.ipynb`
20 cells (12 md / 8 code), 499 lines, **< 5 s total. No PyTorch — pure numpy/scipy/matplotlib.**

| Cell(s) | ~Lines | What it is | Recommendation |
|---|---|---|---|
| **7** | **39 md** | Full 4-step Lagrange-multiplier derivation of the adjoint: augment the loss, integrate by parts, choose `a(t)` to kill the integral, read off `dL/dθ`. Boxed result. | **Compress to ~15 lines.** The mathematical heart of the corpus, but a 2-hour workshop cannot spend 10 minutes on a derivation. Keep the four step *headings* and the boxed adjoint ODE; move the algebra to an appendix or a collapsed cell. |
| **8** | **63** | The signature 3-panel figure: forward `h(t)` / backward `a(t)` with an "Initialized here, solves backward →" annotation / shaded integrand whose area *is* the gradient | **Reuse near-verbatim.** This single figure does more for intuition than the derivation does, and it costs nothing to run. **If you take one thing from this notebook, take this.** |
| **12** | **41** | Hand-drawn 5-box flowchart of the adjoint algorithm, in pure matplotlib | **Reuse.** Renders anywhere, no dependencies, prints well. Worth generalizing into a `draw_flowchart(steps, arrows)` helper. |
| 14 | 57 | Gradient check: adjoint vs analytic vs finite differences, with a log-scale relative-error panel | **Reuse, compress to ~25 lines.** Directly supports brief pitfall 5 and the "finite differences do not scale" argument in §4. |
| 16 | 42 | Memory/compute scaling plot (`state_dim=128`, solver steps 10→5000) | **Reuse or rebuild.** This is the O(1)-memory argument made visually. Small. |
| 11 | 16 md | 4-row table generalizing to vector state; names `torchdiffeq.odeint_adjoint` | **Reuse near-verbatim.** |
| 1 | 14 | The most complete `plt.rcParams` style preamble in the corpus | **Reuse** — promote to `workshop_utils` and use it in every notebook so the figures match. |
| 4, 6, 10 | 37+31+44 | Solution family, tangent-line gradient, 2×3 θ-sweep | **Compress or skip.** Pleasant, redundant with cell 8. |
| 0 | 16 md | Anchor-linked table of contents (`<a id=...>`) | **Skip** — the only notebook using this device; do not spread it. |

**Re-symboling required.** `h(t)` → **`u(t)`** (state), `θ` → **`φ`** (NN weights; here it is a
scalar, so say so), `T` → **`t_end`** (terminal time). The adjoint variable `a(t)` **collides with the
workshop's `a` for static attributes** — rename the adjoint to **`λ(t)`**, which also matches
`2_simple_hybrid_model`'s own usage and standard optimal-control notation. Do this consistently; it is
the most likely place for a silent notation clash across two agents' notebooks.

---

## 4. `2_simple_hybrid_model.ipynb` — the closest thing to the workshop's target

`/home/andrbenn/workspace/differentiable_modeling_workshop/refs/computational_methods_course/notebooks/hybrid_modeling/2_simple_hybrid_model.ipynb`
22 cells (8 md / 14 code), 705 lines, ~3–6 min.

This notebook is already *structurally* what a workshop hybrid-modeling notebook should be:
a conceptual bucket, a learned residual inside the ODE, an honest train/val/test split, three
integration strategies compared, and a gradient check. Mine it heavily.

| Cell(s) | ~Lines | What it is | Recommendation |
|---|---|---|---|
| **7** | **51** | `class HybridBucketCore(nn.Module)` — `unpack_params`, `rhs_bucket(s,u,theta)`, hand-coded `drhs_ds`, `discharge_from_state`, `qhat_normalized`, `qhat_mmday`. A 9-element `raw_theta` parameter. | **Reuse the structure, rewrite the parameterization.** The separation of RHS / observation operator / parameter unpacking is exactly right. But the 9-vector that concatenates physical and neural parameters is the source of the notebook's `θ` ambiguity — **split it into `θ` (physical) and `φ` (neural) as separate `nn.Parameter`s / submodules.** This is the single most important re-symboling in the corpus. |
| **6** | **27 md** | Conceptual bucket → why it is too rigid → the tanh residual → softplus positivity | **Reuse near-verbatim.** The clearest short motivation for a hybrid model in the presenter's voice. |
| 9 | 54 | `RK4HybridModel` and `ImplicitEulerHybridModel` (`dt=1.0`, **`n_iter=30`**) | **Compress: keep RK4, cut implicit Euler or drop `n_iter` to 5–10.** Implicit Euler at `n_iter=30` is the slowest thing here (~345k scalar autograd ops over 8 epochs, 1–3 min on its own) and it teaches a numerics point, not a hybrid-modeling point. |
| **12** | **107** | `class RK4AdjointSolve(torch.autograd.Function)` with explicit `forward`/`backward` | **Reuse if and only if you are teaching the adjoint in code.** The only hand-written `autograd.Function` in the corpus and the most concrete possible demonstration of DTO vs. the adjoint. But 107 lines and a `backward` doing 3 `autograd.grad` calls per timestep. **For a 2-hour workshop: show cell 13's gradient check and the *result*, and put cell 12 in an appendix.** |
| **13** | **12** | Gradient check: direct autograd vs adjoint, prints max abs diff | **Reuse verbatim.** Twelve lines, a few seconds, and it is the whole "are these the same gradient?" question answered empirically. High value per line. |
| **11** | **112 md** | Continuous adjoint derivation with observation-time jump conditions `λ(t_i^-) = λ(t_i^+) + ∂ℓ_i/∂S_i`, then the discrete one-step-map version | **Compress hard, to ~20 lines, or skip.** The longest markdown cell in the corpus. The jump-condition material is genuinely novel relative to notebook 1 but is graduate-numerics depth. |
| 15 | 76 | `mse_by_split`, `denorm_q`, `nse`, `build_model(method)`, `fit_model(method, n_epochs=8, lr=0.03)` | **Reuse `fit_model` as the template** — it returns a well-designed dict (model / theta / history DataFrame / pred_norm / pred_mm / obs_mm / nse_test / rmse_test / states). But it is full-batch with no early stopping; consider merging with `1_lstm_leaf_river`'s `train_model` (§7). |
| 16 | 19 | Runs all three trainings + summary DataFrame | **Cut to one method** (RK4) for the live run; present the other two as precomputed results. **This is the notebook's 2–5 minute cell.** |
| 3, 5 | 20+20 | Leaf River CSV load, 60/20/20 chronological split, train-only normalization, `log1p(Q)` target | **Rewrite for `minicamels`** but **keep the logic verbatim**: chronological split, train-only statistics, `log1p` on streamflow. All three are brief pitfall 6. |
| 4, 18 | 19+15 | Forcing/flow overview and 3-row hydrograph, both with `axvspan` gold/seagreen split shading | **Reuse** — promote to a single `plot_hydrograph_with_splits()` in `workshop_utils`. |
| 19 | 6 | Learned-parameter comparison table across methods | **Reuse.** Cheap and makes the point that different gradient routes reach the same parameters. |
| 0, 2 | 29+3 md | Intro, and the explicit runtime disclaimer | **Reuse the disclaimer habit.** Cell 0 literally says *"To keep the notebook fast enough for classroom use, we train on a contiguous 720-day segment… To use the full record, change `N_TOTAL = len(df_full)`."* Do this in every workshop notebook — state the budget in prose. |

**Re-symboling required.** `S(t)` → **`u`** (or keep `S` as a named scalar state and say
`u = [S]` once — a single-bucket model reads better with `S`, and the brief permits naming components
of `u`). `u` currently means the **forcing** tuple `(precip, temp, p_norm, t_norm)` — **this directly
inverts the workshop scheme and must be renamed to `x`.** `T` means temperature here, which is
workshop-compliant, but check no cell reuses it for a horizon. `θ` → split into `θ` (physical:
`a, k, c, S_0`) and `φ` (neural: `w_r, w_s, w_p, w_t, b`). `r_θ(S,P,T)` → **`NN_φ(u, x)`**. The adjoint
`λ` already matches.

---

## 5. `3_richards_equation_pinn_hybrid.ipynb` — the problem child

`/home/andrbenn/workspace/differentiable_modeling_workshop/refs/computational_methods_course/notebooks/hybrid_modeling/3_richards_equation_pinn_hybrid.ipynb`
21 cells (8 md / 13 code), 1007 lines, **~20–60+ minutes.**

**Verdict: do not run this notebook live in any form.** It is the only place in the corpus where PINNs
and learned constitutive relations appear, so its *content* is needed — but every expensive cell must
be precomputed, shrunk, or replaced with cached weights.

### The four expensive cells, with the specific reason each is slow

| Cell | Knob | Est. CPU time | Why it is slow | Fix |
|---|---|---|---|---|
| **5** | `n_steps=3600`, `MAX_ITER=50`, plus **a `for k in range(ni)` Python loop over 98 interior nodes** inside `picard_step` | **30 s – 2 min** | Tridiagonal assembly is a scalar Python loop, not numpy slicing | **Vectorize the assembly** — mechanical, ~10 lines, buys a 10–50× speedup. Or precompute `h_ref` to a `.npz` and ship it. |
| **11** | **`N_EPOCHS_PINN = 5000`** | **3–10 min** | Each epoch does **two nested `torch.autograd.grad(..., create_graph=True)` calls** (first derivatives, then `d_flux_dz`) and backprops through the second-order graph | **Ship cached weights.** Show a short 200-epoch live run so participants see the loss move, then load the trained state dict. Cutting to 500–1000 epochs also works but degrades the result. |
| **15** | **`N_EPOCHS = 2000`** × `UNROLL_STEPS=8` × `PICARD_ITERS=3` = 24 `picard_step_learned` calls/epoch, each running `thomas_solve_torch` | **10–50 min — the worst cell in the corpus** | `thomas_solve_torch` is a **pure-Python forward-sweep + back-substitution over n=98 scalar tensors** (~200 autograd nodes per solve), so ~4,800 scalar autograd ops are *built and backpropagated* per epoch | **Vectorize `thomas_solve_torch` first** (or batch it), then cut to 200–400 epochs, then ship cached weights. Note the author already tuned this *down*: a code comment in cell 15 explains that a 240-node-deep graph means "gradients vanish before reaching the network weights," which is why `UNROLL_STEPS=8`. |
| **17** | `n_steps=3600`, `MAX_ITER=50`, same Python-loop Thomas solve (under `no_grad`) | **several min – tens of min** | Same scalar-loop problem | Precompute and ship the result array. |

### Assets worth taking

| Cell | ~Lines | What | Recommendation |
|---|---|---|---|
| **14** | **32** (of 103) | `thomas_solve_torch(lower, diag, upper, rhs)` — differentiable tridiagonal solve, with a docstring explaining *why* it is out-of-place (in-place assignment breaks autograd) | **Reuse the docstring's lesson verbatim; rewrite the implementation.** The docstring is brief pitfall 3 in the presenter's own words. The implementation must be vectorized before it appears in a live notebook. |
| **14** | ~35 | `class ConstitutiveNet(nn.Module)` — 3-head output `(θ̂, K̂, Ĉ)` with sigmoid/exp/softplus **physical constraints on each head** | **Reuse near-verbatim, re-symbol.** This is the cleanest example of brief pitfall 4 (constrain NN outputs to physical ranges) in the corpus after chapter 12's `HydroParam`. |
| **13** | 22 md | *"what if we don't know θ(h) or K(h)? …laboratory measurements of soil water retention curves are expensive, spatially variable, and often unavailable"* + the `### Training signal` / `### What we keep (physics)` / `### What we learn (data-driven)` triad | **Reuse near-verbatim.** The best short statement of the learned-closure idea in the corpus, and the three-heading structure is a template worth adopting for every hybrid model the workshop presents. |
| **10** | 43 | `pinn_losses(model, n_col=2000, n_bc=300, n_ic=200)` — the IC/BC/PDE composite loss, including the `K_REF` residual normalization and **square-root-biased time sampling** `t_col = (u_t**2 * T_end)` to densify early times | **Reuse.** Self-contained, transferable to any PINN, and contains two non-obvious tricks worth calling out explicitly. |
| **9** | 68 | Torch mirrors `Se_torch/theta_torch/K_torch/C_torch`, plus `class PINN` with a **scaled-sigmoid output clamp to `[h_bot, h_top]`**. `C_torch` carries a docstring explaining it is analytic *to avoid a `.backward()` conflict with `create_graph=True`* | **Reuse `PINN` and the clamp; compress the constitutive mirrors.** The `C_torch` docstring is a genuine gotcha worth a callout. |
| **16** | 41 | Learned vs true constitutive curves, 3 panels, with an extrapolation caveat printed | **Reuse near-verbatim.** This is the "did we recover the hidden function?" figure — the workshop's most persuasive possible visual, and structurally identical to chapter 12 cell 46. |
| 4 | 54 md | Celia–Zarba modified-Picard discretization, the Thomas algorithm, and the Celia et al. (1990) benchmark setup (0.6 m column, 100 layers, 60 h) | **Compress to ~15 lines.** Cite `celia1990general`; nobody needs the full derivation in 120 minutes. |
| 2 | 29 md | Van Genuchten relations, one `###` subsection per equation | **Compress.** Keep the three equations, drop the subsection scaffolding. |
| 18 | 130 | Three-figure comparison block (profiles coloured by time / styled by method; pointwise abs error; RMSE bar chart) | **Compress heavily or rebuild.** The most polished plotting in the corpus, but monolithic; the `mlines.Line2D` proxy-legend idiom is the part worth keeping. |
| 0 | 21 md | `## Motivation` / `## What we will cover` / `## Earth Science Connection` opening triad | **Reuse as a template** — shared with `rnns/0` and `rnns/1`; the most consistent structural convention in the course material. |

**Re-symboling required — this notebook has the worst collisions in the corpus.**
`θ` means **volumetric water content** here *and* PINN weights in `ĥ(z,t;θ)`. Per brief §1b: water
content → **`θ_w`** (or `ϑ`), with the collision named once in prose since `θ` for water content is
too entrenched in soil physics to rename silently; PINN weights → **`φ`**. `h` (pressure head) →
**`ψ`**, freeing `h` for hidden state elsewhere in the workshop. `C(h)` (specific moisture capacity)
stays `C` but note the workshop's LSTM cell state is `c` — keep the case distinction strict. `z` is
already depth and already compliant. `f_φ` for the constitutive net is **already compliant** with the
workshop scheme and should be the model other notebooks follow.

---

## 6. `chapter_12_ai_for_physics_inspired_hydrology_modeling.ipynb` — the voice, and the small parts

`/home/andrbenn/workspace/differentiable_modeling_workshop/refs/chapter_12_ai_for_physics_inspired_hydrology_modeling.ipynb`
89 cells (45 md / 44 code), 1271 lines, Python 3.8.12. **Halts at cell 67 (missing data file).**

This is the presenter's own published book chapter and the model for the workshop's voice. Its
hydrologic content, however, is built on a CAMELS NetCDF that is not in this repo — so the *data half*
must be rebuilt on `minicamels` regardless.

**What it actually implements** (in case an agent assumed otherwise): a linear reservoir
`dS/dt = kS`; a nonlinear reservoir with `K(x) = -0.1·tanh(10(x-0.5))` learned by an MLP; and a
**two-bucket conceptual catchment model** (surface `S_0`, subsurface `S_1`) with ET, drainage,
saturation-excess surface flow and subsurface flow, with flux forms taken from FUSE.
**There is no SNOW-17 and no snow at all** — the chapter explicitly filters snow-dominated basins out.

### Assets worth taking

| Cell | ~Lines | What | Recommendation |
|---|---|---|---|
| **51** | **15** | `class HydroParam(nn.Module)` — `__init__(self, low, high, net)`, `forward(x) = range*sigmoid(net(x)) + low`, with `low`/`high`/`range` as non-trainable `register_buffer`s | **Reuse near-verbatim. This is the highest-value single class in the corpus.** It is brief pitfall 4 made executable, and it is exactly what Lamichhane's sigmoid range-map does. Every learned-parameter notebook should import this. |
| **34** | ~30 | `class MLP(width, depth, activation=nn.Tanh, in_dim=1, out_dim=1, bias=True, linear=nn.Linear)` — generic dense stack, `depth+2` layers, `nn.ModuleList` | **Reuse near-verbatim.** The most portable building block; pairs with `HydroParam`. |
| **53–61** | 15/17/18/10/16 | `ETTerm`, `DrainageTerm`, `SaturatedAreaTerm`, `SurfaceFlowTerm`, `SubsurfaceFlowTerm` — one `nn.Module` per flux, each clamped to a physically valid range | **Reuse the pattern; audit the code.** One-module-per-flux is an excellent teaching structure and maps directly onto the brief's taxonomy (swap any single module for `NN_φ`). But see the defect list below. |
| **52–62** | 10/9/9/7/9 md | One markdown cell per flux term, strictly alternating equation-then-class | **Reuse the alternation as a structural convention.** It is the clearest exposition pattern in the presenter's corpus. |
| **1** | 28 md | The taxonomy cell: disentangles KGML / NeuralODE / PINN / Hybrid, splits hybrids into three camps, ~25 citations | **Compress and reconcile with brief §2.** Valuable because it is the presenter's own framing, but the brief's §2 is more precise about PINNs and about the PGML boundary. Where they differ, the brief wins; where the chapter's phrasing is better, take the phrasing. |
| **18** | 39 md | Numerical optimization: the sustained **valley/topography analogy** carried through learning rate and momentum | **Reuse near-verbatim.** The best analogy in the corpus and the right level for a non-DL-expert audience. |
| **5, 6** | 36+13 md | Autodiff theory: worked chain rule on `sin(x²)+cos(x²)`, forward-mode dual numbers with a numeric walkthrough at `x=π`, then reverse mode | **Compress to ~15 lines, and fix the missing image.** Cell 5 embeds `assets/computational_graph.png`, **which does not exist** — either regenerate the figure or drop the embed. |
| **39** | ~5 md | *"You might be thinking 10 samples is way too small… but we have a very strong inductive bias… we've directly encoded the differential equation and are only attempting to figure out a parameterization of it."* | **Reuse verbatim.** This is the data-efficiency argument in one sentence, and it answers the question a hydrologist will actually ask. |
| **33** | 3 md | *"you can't directly measure the conductivity of the 'reservoir' but you can measure storage levels"* | **Reuse verbatim.** The setup for every learned-closure example. |
| **46** | 19 | `plot_conductivity(model, v)` — target curve + NN curve + training-point diamonds | **Reuse.** The "did the network recover the hidden function?" plot; identical in spirit to `3_richards` cell 16. **Use this figure as the payoff of the synthetic-truth twin experiment** the brief recommends. |
| **19, 21, 44** | 10/17/6 | `newton_solve`, `plot_newton_solve`, `plot_loss` | **Reuse `plot_loss`; unify the two `newton_solve` copies** (the other is a method on `NeuralReservoir`, cell 34, with an extra `err` guard). |
| **69** | 23 | `class MultipleTrajectoryDataset(Dataset)` — chops an xarray Dataset into fixed-length trajectories for multiple-shooting | **Reuse the idea; rewrite for `minicamels`.** Multiple shooting (cell 68's markdown explains it, analogized to RNN training) is the right answer to brief pitfall 1. |
| **86** | 21 md | `### 6. Conclusions` + 5 numbered assignments + open questions | **Reuse for Agent 4.** The only explicit exercise block in the corpus, and directly useful for the discussion segment. |
| **87, 88** | 74 md | 40-entry reference list | **Cross-check against `references.bib`**, then drop; the workshop has its own bibliography. |
| 43, 49, 77 | 4/11/14 | The three training loops (`max_epochs=300` ×2; `6 × 7 × 2` nested) | **Shrink aggressively.** Cell 77 does ≈46,000 `odeint` calls (126 model runs × 365 daily steps), 2–10 min. Cells 43 and 49 each call `torch.autograd.functional.jacobian` on a 6-layer MLP inside every Newton iteration — 1–5 min each. |
| 67 | 12 | `xr.open_dataset('./data/camels_data.nc')` | **Skip — replace with `minicamels`.** File missing; xarray APIs downstream are deprecated. |
| 2 | 9 | `mpl.rcParams['figure.dpi'] = 300` | **Change to 100.** A print setting; at 300 every figure is slow and large. |

### Four defects in chapter 12 — do not propagate these

1. **Cell 75 swaps `b` and `c`.** The call is
   `HydroSimulator(S0max, S1max, p, ku, ks, b, c, n)` but the signature (cells 63/65) is
   `(..., p, ku, ks, c, b, n)`. The drainage exponent gets the saturation-area bounds and vice versa.
2. **Cell 63 draws ET from the wrong bucket**: `self.et = self.et_term(x, S1, pet)` uses the
   *subsurface* store, while the prose and the `dS0/dt` equation both say ET comes from the surface
   store `S_0`.
3. **Cell 65 truncates BPTT to one day**: `storage = storage[-1].clone().detach()` inside the daily
   loop means gradients never flow across timesteps within a trajectory. Whether intentional or not,
   it materially changes what "training through the ODE" means, and it should be either fixed or
   explained.
4. **Cell 85 uses an undefined `i`**: `nse(q_pred[i:], q_true[i::])`.

Also: `class SubsurfaceFlowTerm` (cell 61) stores `self.z` as a plain tensor rather than a
`register_buffer`, unlike its siblings — a latent `.to(device)` bug.

**Re-symboling required.** Chapter 12 avoids `θ` entirely, which makes it the *easiest* to port.
`S`, `S_0`, `S_1` → components of **`u`** (keeping the `S` names is fine and reads better). `x` is
used for static attributes → rename to **`a`**; forcings `P`, `PET` → components of **`x`**. `k`
(reservoir conductivity, **negative**) collides with `2_simple_hybrid_model`'s `k` (recession
coefficient, **positive**), and `c` means different exponents in the two notebooks — **pick one bucket
model for the workshop and do not mix their symbols.** The eight `HydroParam` outputs are the
workshop's `θ`, and the MLPs producing them carry `φ`.

---

## 7. `rnns/` — needed only if a notebook uses an LSTM

### `0_rnn_theory.ipynb`
`/home/andrbenn/workspace/differentiable_modeling_workshop/refs/computational_methods_course/notebooks/rnns/0_rnn_theory.ipynb`
13 cells (8 md / 5 code), 392 lines, ~1–2 min.

| Cell | ~Lines | What | Recommendation |
|---|---|---|---|
| **6** | **18 md** | BPTT as a product of Jacobians; the singular-value argument for vanishing/exploding gradients; clipping | **Reuse near-verbatim.** This is brief pitfall 1, and it is the concept that transfers *directly* to unrolled hydrologic simulation — say that explicitly: a 10,957-step daily run is the same object as a 10,957-step RNN. |
| **7** | 47 md | LSTM: the "conveyor belt" analogy, a 3-row gate table, all six equations, and `### Why does this help?` explaining the additive path | **Compress to ~20 lines** unless an LSTM is central. The additive-path explanation is the part to keep. |
| **3** | 40 | Hand-drawn unrolled-RNN schematic in matplotlib (`FancyBboxPatch`, coloured arrows, `axis('off')`) | **Reuse.** The corpus's best pedagogical schematic. |
| **8** | 70 | LSTM gate-activation visualization on synthetic rainfall, with **manual gate extraction from `weight_ih_l0`/`weight_hh_l0`** (slice `[:H]`, `[H:2H]`, `[2H:3H]`) | **Reuse the extraction idiom** as `extract_lstm_gates(lstm, x)`. The only place in the corpus that opens up `nn.LSTM`'s internals; high value if the workshop shows what an LSTM-emitted parameter series looks like over a snow season. |
| 11 | 80 | RNN/GRU/LSTM comparison on a synthetic copy task (`n_epochs=40`, three models, ~3,000 optimizer steps) | **Skip or precompute.** 30 s – 2 min, and it teaches an ML point, not a hydrology point. |
| 9 | 38 md | GRU equations + a 5-row LSTM-vs-GRU practice table | **Skip.** |

**Re-symboling.** `h_t`, `c_t`, `x_t` are the LSTM's own conventions and are **already compliant** —
`h`/`c` hidden and cell state, `x` input. The one hazard: `T` here means sequence horizon; rename to
**`L`** (lookback/sequence length) to free `T` for temperature.

### `1_lstm_leaf_river.ipynb`
`/home/andrbenn/workspace/differentiable_modeling_workshop/refs/computational_methods_course/notebooks/rnns/1_lstm_leaf_river.ipynb`
25 cells (12 md / 13 code), 482 lines, **~10–40 min — over budget as written.**

| Cell | ~Lines | What | Recommendation |
|---|---|---|---|
| **8** | **19** | `class HydrologyDataset(Dataset)` — `__init__(self, x, q, lookback)`, returns `([L,F], [1])` | **Reuse near-verbatim** (rewrite the input for `minicamels`). The cleanest windowed-timeseries Dataset in the corpus. |
| **12** | **50** | `train_model(model, train_dl, val_dl, n_epochs=30, lr=1e-3, patience=8)` — train/val loop, gradient clipping, **best-state tracking and restore, early stopping**, history dict | **Reuse — this is the best general training harness in the corpus.** Merge with `2_simple_hybrid_model`'s `fit_model` (which has better return structure but no early stopping) and promote one version to `workshop_utils`. |
| **5** | 9 md | *"For time series we **never shuffle**: past cannot be used to predict past… We normalise inputs using the **training set statistics only** to avoid data leakage. Streamflow is log-transformed…"* | **Reuse verbatim.** Brief pitfall 6 in three sentences, better phrased than the brief phrases it. |
| **14** | 17 md | RMSE and NSE formulas + a **5-row NSE interpretation table** (1.0 perfect / >0.75 good / 0.5–0.75 satisfactory / <0.5 unsatisfactory / ≤0 worse than climatology) | **Reuse verbatim.** The best "how to read this number" table in the corpus, and exactly what a mixed-background audience needs. |
| **15** | 35 | `predict(model, dl)`, `denorm_streamflow`, `calc_metrics(true, pred)` → RMSE / NSE / PBias | **Reuse `calc_metrics` as the canonical metric helper** — it is the richest of the three competing versions. Settle the argument order (recommend `(obs, sim)`) and delete the others. |
| **17, 18** | 29+22 | Annotated 3-row hydrograph (NSE/RMSE in each panel title) and 1:1 obs-vs-pred scatter grid | **Reuse near-verbatim.** Copy-ready. |
| **22** | 39 | Hidden-state probe: top-3 highest-variance hidden dimensions plotted against precipitation and flow | **Reuse if the workshop makes an interpretability point.** Pairs naturally with the hybrid argument: the LSTM's hidden state *resembles* soil moisture; a hybrid's state *is* soil moisture. |
| **10** | 59 | `MLPBaseline`, `LSTMModel`, `GRUModel`, `count_params` | **Compress to LSTM + `count_params`.** `count_params` supports the parameter-count argument in brief §4 directly — use it to print "SNOW-17: 8 parameters. This LSTM: 19,208." live. |
| 12 (call site) | — | `n_epochs=60` for three models | **Cut to 10–15 epochs, one model.** 6–20 min as written. |
| **20** | **33** | Lookback sensitivity sweep: **7 full retrainings** at `lookback_values=[7,14,30,60,120,180,365]` | **Skip or precompute.** 5–20 min. Scientifically interesting (catchment memory as an empirical question) but the workshop cannot afford it. Ship the result as a static figure. |
| 3 | 8 | Reads the **full 10,959-day** Leaf River record | **Note the inconsistency:** `2_simple_hybrid_model` deliberately truncates to 720 days and this notebook does not. Pick one convention. |

---

## 8. Consolidated recommendation: what to put in `workshop_utils/`

The package is already declared in `pyproject.toml` but does not exist yet. Building it first will
save all three notebook agents from duplicating and diverging.

| Proposed name | Source | ~Lines |
|---|---|---|
| `rk4_step`, `odeint_fixed` | `0_neural_odes` cell 6 | 15 |
| `HydroParam(low, high, net)` | `chapter_12` cell 51 | 15 |
| `MLP(width, depth, ...)` | `chapter_12` cell 34 | 30 |
| `calc_metrics(obs, sim)` → NSE / RMSE / PBias / KGE | `1_lstm_leaf_river` cell 15, extended with KGE | 20 |
| `train_model(...)` with early stopping + best-state restore | `1_lstm_leaf_river` cell 12 merged with `2_simple_hybrid` `fit_model` | 60 |
| `HydrologyDataset(x, q, lookback)` | `1_lstm_leaf_river` cell 8 | 20 |
| `plot_hydrograph_with_splits(...)` | `2_simple_hybrid_model` cells 4, 18 | 30 |
| `plot_learned_vs_true(...)` | `chapter_12` cell 46 + `3_richards` cell 16 | 25 |
| `plot_loss(history)` | `chapter_12` cell 44 | 10 |
| `set_plot_style()` | `1_neural_ode_adjoint` cell 1 | 15 |
| `smooth_min`, `soft_clamp`, `smooth_threshold(k=20)` | **new** — implements brief pitfall 3; the reference forms are in `3_richards` cell 14 and Kalauni | 20 |
| `load_minicamels_tensors(basin_ids, ...)` | **new** — chronological split, train-only normalization, `log1p(Q)` | 50 |

---

## 9. The presenter's voice — two registers, pick deliberately

The corpus contains two distinct voices, and the workshop should choose one rather than drifting
between them.

**The course-notebook register** (`0`–`3`, `rnns/`): first-person plural "we", essentially never
second person, no "Let's" (two exceptions across five notebooks), no exercises, no callouts. Math is
stated with explicit shapes. Structural habits worth adopting: the
`## Motivation` / `## What we'll cover` / `## Earth science connection` opening triad; summary tables
instead of bulleted recaps; **explicit cross-references to sibling notebooks by filename**; and — the
most distinctive habit — **naming its own simplifications and runtime budget in prose** (`0_neural_odes`
cell 16's "what this notebook simplifies" table; `2_simple_hybrid_model` cell 0's "to keep the
notebook fast enough for classroom use…"). `3_richards` adds a habit worth stealing: **explaining the
numerical trap in a code docstring**, right where someone would hit it.

**The chapter-12 register**: constant second person ("you might be thinking", "if you're ambitious"),
"Let's" everywhere (`### Let's train!` is a literal heading), conversational hedges ("*most of the
time it works*", "Let's call this good enough and move on"), sustained analogies (optimization as
topography), inline exercises woven into prose, and `> ### A note on <topic>` blockquote callouts.
Math is narrated rather than derived, with explicit permission to skip. It also uses
**confidence-building as method**: every new tool is validated against a known analytic answer before
being used in anger.

**Recommendation for a 2-hour workshop with practising modellers who are not DL experts: use the
chapter-12 register.** It is warmer, it is the presenter's own published voice, it grants permission
to not follow every derivation, and its validate-against-a-known-answer habit is exactly right for an
audience being asked to trust gradients through their own model. Take the course notebooks' structural
discipline — the opening triad, the summary tables, the stated runtime budgets, the honesty tables —
and write it in chapter 12's voice. Map `> ### A note on X` onto whichever admonition convention the
workshop settles on, and settle that once.

One housekeeping note: the corpus mixes British spelling (`normalise`, `visualise` in `1_lstm_leaf_river`
and `3_richards`) with American (`chapter_12`). Pick one.

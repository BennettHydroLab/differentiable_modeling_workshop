# Annotated Reading List

*For participants of "Differentiable & Hybrid Modeling for Hydrologic Systems."*

Two hours is not enough time to learn this field, only enough to see the shape of it. This list is
what to read next, tiered by how much you want to invest. Citation keys refer to
`references.bib` in this repository.

A note on effort: the "Start here" tier is about four hours of reading and will get you to the point
of holding an informed opinion. The "Go deeper" tier is a semester.

---

## Start here

**Shen et al. (2023), *Differentiable modelling to unify machine learning and physical models for
geosciences*, Nature Reviews Earth & Environment.** `shen2023differentiable`
The framing paper for the whole field, and the one to read first even though it is a review rather
than a method. Its central claim is deceptively simple: the thing that makes deep learning work is
not the neural network, it is *differentiable programming*, and once your physical model is
differentiable too the boundary between the two disappears. Read the Introduction, "The root of deep
network's success," and "Differentiable Geosciences" sections carefully; skim the domain survey.
Everyone should read this. Note that the freely available arXiv version has the method taxonomy and
the challenges section redacted — get the published version if you can.

**Rackauckas et al. (2021), *Universal Differential Equations for Scientific Machine Learning*.**
`rackauckas2021universal`
The mathematical framing: a UDE is a differential equation "defined in full or in part by a universal
approximator." Read Sections 1, 2.1, 2.3 and the Discussion; the rest is a tour of the Julia SciML
ecosystem and the supplements are reference material. Worth it for two things in particular: the
Lotka–Volterra example, where a network fills in a missing term and symbolic regression then reads
the term back out as an equation; and Supplement S6–S8, which is the clearest published account of
the trade-offs among adjoint methods. Read it even if you will never write Julia — the formalism
transfers, the software does not.

**Höge, Scheidegger, Baity-Jesi, Albert & Fenicia (2022), *Improving hydrologic models for
predictions and process understanding using neural ODEs*, HESS.** `hoge2022improving`
The cleanest hydrologic entry point, and the one to read if Shen felt too abstract. A conceptual
bucket model with its internal process equations replaced by neural networks, built and evaluated
with the care hydrologists expect. Open access. If you only read one *method* paper, read this one.

**Kratzert et al. (2018), *Rainfall–runoff modelling using LSTM networks*, HESS.**
`kratzert2018rainfall`
The baseline every hybrid model is measured against, and still the clearest statement of why deep
learning works in rainfall–runoff. Pair it with Kratzert et al. (2019, EA-LSTM)
`kratzert2019ealstm`, which shows how static catchment attributes let one regional model behave
differently in different basins — the trick that makes prediction in ungauged basins work, and the
same trick every differentiable parameter-learning scheme uses. Read these if you have not yet made
peace with the LSTM literature; skip if you have.

**Tsai et al. (2021), *From calibration to parameter learning*, Nature Communications.**
`tsai2021calibration`
The differentiable parameter learning (dPL) paper. A neural network maps catchment attributes to VIC
parameters and is trained *through* VIC against observations at all sites simultaneously. This is the
paper to hand a colleague who says "we already calibrate our model, why do I need this" — the answer
is the global loss, the spatial coherence of the resulting parameter fields, and the collapse of a
100-CPU-cluster job into a GPU-hour. Short and readable.

**Lamichhane & Bennett (2025), *Dynamic Parameterization of SNOW-17 using LSTM*.**
`lamichhane2025dynamic`
The presenter's own snow paper, and the source of the workshop's central example. 734 SNOTEL sites;
an LSTM emits SNOW-17's parameters at every timestep; the hybrid matches a pure LSTM on KGE but beats
it by roughly two weeks on melt-out timing. Read it for the evaluation design as much as the method:
it is a worked demonstration that an aggregate score can hide the difference that actually matters
operationally. Preprint — check for the published version.

---

## Go deeper

**Kalauni, Gupta & Bennett (2026, in review), *Hybrid models can serve as diagnostic tools to probe
land surface parameterizations*.** `kalauni2025hybrid`
The presenter's other paper, and the most intellectually ambitious thing on this list. It builds one
differentiable land surface model in three versions — pure physics, learn-the-resistances,
learn-the-fluxes — and asks how much physics you actually need. The answer is subtle: more neural
flexibility wins per-site, but the *more constrained* version generalizes to unseen sites just as
well with less bias. The diagnostic argument in Section 3.3 is the payoff: the learned model recovers
the decline of latent heat at high vapour pressure deficit that the physics structurally cannot
produce, which tells you precisely what to fix in the parameterization. For anyone who wants to use
hybrid models to *learn something*, not just to predict better.

**Chen, Rubanova, Bettencourt & Duvenaud (2018), *Neural Ordinary Differential Equations*, NeurIPS.**
`chen2018neural`
The origin of the neural ODE and of `torchdiffeq`. Read Sections 1–3 for the adjoint derivation; the
normalizing-flows material is a different subject. Worth reading in the original because almost every
secondary account garbles the memory argument. For most hydrologic problems you will not use the
continuous adjoint — but you should understand what you are choosing not to use, and why.

**Kidger (2022), *On Neural Differential Equations*.** `kidger2022neural`
A 200-page doctoral thesis that is, unexpectedly, the best textbook in the area. Chapters 1–5 cover
neural ODEs, CDEs, SDEs and the numerics with unusual care about what actually goes wrong. If you
find yourself fighting a solver, this is where the answer is. Also the origin of Diffrax. Read
selectively; do not attempt front to back.

**Feng, Liu, Lawson & Shen (2022), *Differentiable, learnable, regionalized process-based models…*,
WRR.** `feng2022differentiable`
δHBV: a conceptual model on PyTorch with neural parameterization, reaching median NSE 0.732 against a
pure LSTM's 0.748 on CAMELS — while also producing evapotranspiration and baseflow that nobody
trained it on. The concrete demonstration that you can have ML accuracy and internal physical
variables at the same time. Its companion `feng2023suitability` extends the argument to ungauged
regions and climate projection; read that one if PUB or nonstationarity is your problem.

**Kavetski & Clark (2010) and Clark & Kavetski (2010), *Ancient numerical daemons of conceptual
hydrological modeling*, parts 1 and 2, WRR.** `kavetski2010ancient`, `clark2010ancient`
Written years before anyone said "differentiable," and now more relevant than when published. The
argument is that sloppy time stepping in conceptual models produces a jagged objective surface that
wrecks calibration. Everything they say about calibration applies, with more force, to gradient
descent — a gradient of a discontinuous objective is worse than useless because it looks fine. Read
these before you write your own differentiable bucket model.

**Knoben et al. (2019), *MARRMoT v1.2*, GMD.** `knoben2019marrmot`
Forty-six conceptual hydrologic models rewritten as smooth, continuous state-space formulations. The
best single reference for *how* to replace a threshold with something differentiable, from people who
did it 46 times. Use it as a cookbook rather than reading it through.

**Best et al. (2015), *The plumbing of land surface models*, JHM**, and **Abramowitz et al. (2024),
PLUMBER2, Biogeosciences.** `best2015plumbing`, `abramowitz2024plumber2`
The uncomfortable result that motivates a lot of this work: simple empirical models outperform
sophisticated land surface models at predicting surface fluxes, and the finding survived scaling up
to 22 models and 170 sites. If you want to know why anyone would replace working physics with a
neural network, this is the reason.

**Beven (2006), *A manifesto for the equifinality thesis*, J. Hydrology.** `beven2006manifesto`
Read it, or re-read it, with differentiable modeling in mind. Nothing about gradients makes
equifinality go away; letting parameters vary in time makes it dramatically worse. This is the
counterweight to the optimism in everything above.

**Karniadakis et al. (2021), *Physics-informed machine learning*, Nature Reviews Physics.**
`karniadakis2021physics`
A map of the wider physics-informed ML landscape, useful mainly for placing PINNs, operator learning
and hybrid models relative to one another so you stop conflating them. Read the taxonomy, skim the
rest.

**Raissi, Perdikaris & Karniadakis (2019), *Physics-informed neural networks*, JCP.**
`raissi2019physics`
The PINN paper. Read it to understand what a PINN actually is — a network trained to *be* a solution
surface, with the PDE enforced through derivatives of the network with respect to its own inputs.
Understanding this precisely is what stops you from calling every hybrid model a PINN. Note the real
constraint: a PINN is tied to one initial/boundary-condition pair and must be retrained for another.

---

## Software & tooling

Honest assessment, as of this workshop. Maturity assessments are the author's judgement, not
measurements.

**PyTorch + a plain `for` loop.** *The default, and what this workshop uses.*
For a daily, fixed-step hydrologic model, do not reach for an ODE library at all. Write the time loop
in Python, let autograd tape it, and call `.backward()`. This is discretize-then-optimize: the
gradient is exact for the computation you actually performed, the debugging story is ordinary
PyTorch, and 10,000 daily steps of a small state vector fits in memory without complaint. The
failure mode is memory on very long or very high-dimensional rollouts — at which point you train on
short windows and carry the state forward, which is what the presenter's papers do anyway. Mature,
boring, correct. Start here.

**`torchdiffeq`.** *Pick it when you need adaptive stepping or want to demonstrate the adjoint.*
Chen et al.'s reference implementation. `odeint` backpropagates through the solver steps;
`odeint_adjoint` uses the continuous adjoint for O(1) memory. Mature enough and widely used, but
narrow: no stiff solvers worth the name, no DAEs, no SDEs, and the reverse-solve adjoint is
documented (by Rackauckas, with benchmarks) to diverge on stiff and on upwinded PDE problems. Fine
for the non-stiff, modest-dimension systems most conceptual hydrology produces. Do not use
`odeint_adjoint` by default — measure first, and check the gradient against plain `odeint`.

**`diffrax` (JAX).** *Pick it if your system is stiff, or if you need speed and `vmap`.*
Patrick Kidger's library, and the most capable differentiable ODE/SDE stack in Python: implicit and
ESDIRK solvers, adaptive stepping, several adjoint modes, and — importantly — it will differentiate
through an implicit solve via the implicit function theorem rather than through the Newton
iterations. This is what Kalauni et al. use, and their stiff soil-heat column is exactly the case
where PyTorch's ecosystem leaves you stranded. Costs: you must learn JAX's functional style, debugging
under `jit` is harder, and you will want `equinox` `kidger2021equinox` for anything resembling a
normal neural network module. Actively maintained, smaller community than PyTorch. Worth the switch
for a research project; not worth it for an afternoon.

**Julia SciML — `DifferentialEquations.jl`, `SciMLSensitivity.jl` (formerly
`DiffEqSensitivity.jl`), `DiffEqFlux.jl`.** *The most complete, and the least likely to match your
group's existing code.*
Rackauckas et al.'s ecosystem, and genuinely the reference implementation of the UDE idea: 300+
solvers, stiff systems, DAEs, DDEs, SDEs, and the full menu of forward and adjoint sensitivity
methods with a decision tree for choosing among them. Their published benchmarks show order-of-
magnitude-plus advantages over `torchdiffeq` on scientific-model-shaped problems — treat these as
authors' benchmarks, but the *feature* gap is not in dispute. The real cost is sociological: if your
lab writes Python, adopting Julia is a multi-month commitment and a hiring constraint. Read the UDE
paper for the ideas regardless; adopt the stack only if you are starting fresh and your problems are
genuinely stiff.

**`NeuralHydrology`.** `kratzert2022neuralhydrology` *Pick it for LSTM baselines, not for hybrids.*
The Kratzert/Google group's PyTorch library for deep learning on CAMELS-style datasets: data loaders,
training loops, evaluation, and reference implementations of LSTM and EA-LSTM. Well maintained, well
documented, and the fastest route to a *credible* LSTM baseline — which matters, because a weak
baseline makes a hybrid model look better than it is. It is not a differentiable-physics framework
and does not pretend to be. Use it for the comparison arm of your study.

**`HydroDL`** (Shen group, Penn State). *Pick it to reproduce the δHBV/dPL line of work.*
The research code behind Feng et al. and Tsai et al. — PyTorch implementations of differentiable HBV
and the dPL machinery. Its value is that it is the actual code behind published, well-benchmarked
results. Its cost is that it is research code: expect thin documentation, expect to read the source,
and expect to adapt rather than configure. Look for the group's more recent packaging efforts
(`hydroDL2` / `generic_deltaModel`) before starting from the original repository, since the project
has been reorganized more than once.

**`MARRMoT`** `knoben2019marrmot` **and `SUMMA`.** *Not differentiable — read them for the physics.*
MARRMoT (MATLAB/Octave) gives you 46 conceptual models already written as smooth continuous
state-space systems, which is a substantial head start if you are about to differentiate one. SUMMA
is the multiple-hypothesis land model the presenter's earlier work `bennett2021deep` built on. Use
both as sources of well-posed model structures to port, not as tools to run.

**`DeepXDE`.** `lu2021deepxde` *Pick it only if you actually want a PINN.*
The reference PINN library. Mature within its scope. But be sure a PINN is what you want — if you
have a working solver and an uncertain closure, a UDE is almost always the better formulation, and
DeepXDE is not built for that.

---

## If you read only three things

`shen2023differentiable` for why, `hoge2022improving` for how it looks in hydrology, and
`lamichhane2025dynamic` for what a careful evaluation of a hybrid model actually involves.

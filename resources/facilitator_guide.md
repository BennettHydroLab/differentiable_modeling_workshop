# Facilitator guide

For the person running the room. Participants do not need this file.

## The one-sentence argument

Everything in the two hours serves one claim: **gradients are not faster than
your calibrator — they are the only thing that still works once the question
mark moves from a handful of numbers to a function.** If a participant leaves
able to say that sentence and explain why, the workshop worked.

## Timing

| Clock | Notebook | What must land |
|---|---|---|
| 0:00-0:10 | `00_setup_and_motivation` | Everyone's environment runs. The scaling argument. |
| 0:10-0:25 | `theory/01_gradients_and_autodiff` | What `.backward()` did. Smooth surrogates. |
| 0:25-0:45 | `theory/02_neural_odes_and_udes` | The five UDE forms. Where PINNs sit. |
| 0:45-1:10 | `applications/03_hybrid_bucket_model` | The first learned closure. Look inside it. |
| 1:10-1:30 | `applications/04_dynamic_parameterization` | theta becomes a function. |
| 1:30-1:45 | `applications/05_hybrid_as_diagnostic` | Hybrids as instruments, not just scores. |
| 1:45-1:57 | `advanced/06_practical_guide` | Why it will not train, and what to do. |
| 1:57-2:00 | `advanced/07_open_questions` | Hand off to discussion. |

**If you are running late**, cut in this order:
1. `05_hybrid_as_diagnostic` — the most self-contained; point at it as reading.
2. The second half of `02_neural_odes_and_udes` (the adjoint derivation).
3. `07_open_questions` — but keep the last five minutes for discussion even so.

**Never cut** notebook 00's scaling argument or notebook 03's "look inside the
learned function" section. Those are the two moments the case is actually made.

## Get setup out of the way first

Have people run notebook 00's first two cells **before you start talking**.
The Colab badge is the fallback for anyone whose local environment fights them
— do not spend workshop time debugging a laptop.

`minicamels` fetches data from GitHub raw URLs at runtime. If conference wifi
is hostile, have participants run `load_basin` for the three example basins
early so the data is cached in their session.

## Set expectations about the numbers

Notebook 00 shows **random search beating gradient descent**, and it does this
deliberately. Do not apologize for it or rush past it. This room contains
people who calibrate models for a living, and a workshop that opens by
overselling gradients will lose them in the first ten minutes. Lean into it:
the honest result is what makes the dimensional argument credible when it lands
two cells later.

The same applies throughout. Where a hybrid model does not clearly beat the
physics baseline, say so and ask the room why.

## Questions you will get, and honest answers

**"Isn't this just curve fitting with extra steps?"**
Partly — and where it is, the notebooks say so. The difference is that the
network sits *inside* a mass balance, so it cannot buy accuracy by violating
conservation, and you can plot what it learned against the process it
replaced. A post-hoc residual correction cannot do either. Notebook 03's
"look inside" section is the concrete answer.

**"How much data do I need?"**
Less than for a pure ML model, because the physics supplies most of the
structure — but there is no clean number, and anyone quoting you one is
guessing. Brief §7 lists this as genuinely open.

**"Will it extrapolate to a changed climate?"**
Nobody knows. Shen et al. argue stronger physical priors should help; the
supporting evidence is preliminary. Do not promise more than that.

**"Why not just use an LSTM?"**
Often you should — for streamflow prediction alone, a well-trained LSTM is a
very strong baseline and the differentiable models roughly match rather than
beat it (Feng et al.: median NSE 0.732 vs 0.748). You reach for a hybrid when
you also want the internal fluxes, want to interrogate the physics, or need to
extrapolate somewhere an LSTM has no business going.

**"My model has an `if` statement in it."**
That is the most practically useful question anyone will ask. Notebook 01 and
`workshop_utils.nn` answer it: smooth surrogates, and the sharpness/accuracy
tradeoff. This is usually what stands between a hybrid model that trains and
one that does not.

## Live failure modes

- **A training cell is slower than advertised.** Every notebook is budgeted
  under 3 minutes on a laptop CPU, but a hot room of laptops on battery is
  slower. All notebooks ship already executed, so you can talk over the stored
  outputs rather than waiting.
- **Someone's numbers differ slightly.** Expected — thread counts and library
  versions change the last digit. Metrics should agree to about two decimals.
- **The Colab install is slow.** It pulls torch. Have people start it during
  the introduction, not when they first need it.

## Closing the room

`07_open_questions` is written to hand off to discussion rather than conclude.
Shen's "seven question types" and Kalauni's how-much-physics spectrum are the
two prompts most likely to get hydrologists arguing productively. Budget the
last five minutes for that and let it overrun the notebook.

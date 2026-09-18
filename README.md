# Differentiable & Hybrid Modeling for Hydrologic Systems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BennettHydroLab/differentiable_modeling_workshop/blob/main/notebooks/00_setup_and_motivation.ipynb)

A hands-on, 2-hour workshop for hydrologic modelers who want to combine
process-based models with machine learning — and keep the physics.

Everything is a Jupyter notebook. Everything runs on a laptop CPU. Every
hands-on example uses real data from 50 CAMELS-US basins via
[`minicamels`](https://github.com/BennettHydroLab/minicamels).

## Who this is for

You model hydrologic systems. You are comfortable in Python and you have
calibrated a conceptual model before. You do **not** need prior deep learning
experience — we build the machine learning up from gradients.

## Run it

### In the browser (no install)

Every notebook carries an "Open in Colab" badge. Click it and run the first
cell; it installs what it needs.

### Locally

```bash
git clone https://github.com/BennettHydroLab/differentiable_modeling_workshop.git
cd differentiable_modeling_workshop
uv sync            # or: pip install -e .
uv run jupyter lab
```

## The schedule

| Time | Block | Notebooks |
|------|-------|-----------|
| 0:00-0:10 | Setup & motivation | `00_setup_and_motivation` |
| 0:10-0:45 | Theory | `01_gradients_and_autodiff`, `02_neural_odes_and_udes` |
| 0:45-1:35 | Applications | `03_hybrid_bucket_model`, `04_dynamic_parameterization`, `05_hybrid_as_diagnostic` |
| 1:35-2:00 | Frontiers | `06_practical_guide`, `07_open_questions` |

## Build the book

```bash
uv run jupyter-book build .
```

## License

Materials released under CC BY 4.0; code under MIT.

# Deployable Conformal Classifier

A hands-on project for learning end-to-end ML deployment workflows — training, conformal prediction, serving, containerization, CI/CD, and monitoring — using a simple tabular classifier as the vehicle.

The goal isn't state-of-the-art accuracy. It's building the full pipeline a research model needs before it can be trusted in production, with calibrated uncertainty (via conformal prediction) as the differentiator.

## Project status

**Current stage:** Step 1 — model training ✅

## Roadmap

- [x] **1. Train a baseline model** — `train.py`
- [ ] **2. Wrap with conformal prediction** — use MAPIE + the calibration split to produce prediction sets with a coverage guarantee
- [ ] **3. Serve via API** — FastAPI `/predict` endpoint returning prediction sets, not just point predictions
- [ ] **4. Containerize** — Dockerfile for reproducible deployment
- [ ] **5. CI/CD** — GitHub Actions to lint, test, and build on every push
- [ ] **6. Deploy** — push to a hosting platform (Render / Fly.io / Hugging Face Spaces)
- [ ] **7. Monitor for drift** — track input distribution shift and empirical coverage over time
- [ ] **8. Retrain trigger** — re-calibrate and redeploy when drift crosses a threshold

## Dataset

[UCI Adult (Census Income)](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data) — binary classification task (income >$50K vs. not), downloaded automatically by `train.py`.

## Setup

```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Usage

```powershell
python train.py --output-dir ./artifacts --seed 42
```

This will:
1. Download and clean the Adult dataset
2. Split it into train / calibration / test sets (the calibration split is reserved for conformal prediction in step 2)
3. Fit a `RandomForestClassifier` inside a preprocessing pipeline
4. Evaluate on the test set
5. Save outputs to `--output-dir`

### Output

```
artifacts/
├── model.joblib     # fitted sklearn pipeline (preprocessing + model)
├── calib.csv        # calibration split — features + target
├── test.csv         # held-out test split — features + target
└── metrics.json     # accuracy, classification report, seed used
```

## Project structure

```
.
├── train.py           # Step 1: training script
├── requirements.txt   # Python dependencies
├── .gitignore
├── venv/              # local virtual environment (not tracked)
└── artifacts/         # generated model + data outputs (not tracked)
```

## Notes

- Written as a plain script rather than a notebook, to build the reproducibility habits (CLI args, seeding, saved artifacts) needed for later automation steps.
- The train/calibration/test three-way split is intentional — conformal prediction methods (step 2) need a held-out calibration set separate from both training and final evaluation.

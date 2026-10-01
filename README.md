# Calibrated Decision Models for LLM Security

This repository is the paper-specific reproducibility artifact for:

> From Detection to Decision: Evaluating Calibrated Decision Models for LLM Security

It reproduces the paper's offline calibration/evaluation outputs from frozen prediction artifacts. It does not run Prompt Guard, Jev, PyChomsky, or any hosted LLM reviewer. No API keys, private SDKs, model weights, raw benchmark prompt text, or local experiment paths are required.

## Scope

Included:

- Calibration and metric implementations for ECE, Brier score, NLL, detection metrics, scalar temperature scaling, RQ1/RQ2 tables, reliability figures, and RQ3 security-deferral curves.
- Frozen split membership as ordered sample IDs.
- Frozen prompts and calibration artifacts used by the paper evaluation.
- Probability-only prediction artifacts needed for offline reproduction.

Excluded:

- Raw TensorTrust, BIPIA, NotInject, and PIDS-Bench prompt records.
- Model inference clients and private service configuration.
- Credentials, `.env` files, package indexes, caches, and local virtual environments.

The anonymous author entry in `CITATION.cff` should be replaced before a non-anonymous public release.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

Run the complete offline reproduction from any directory:

```bash
bash scripts/reproduce_all.sh
```

The script writes regenerated outputs under `reproduced/` and refuses to overwrite an existing `reproduced/` directory. Remove that directory before rerunning.

Expected outputs include:

- `reproduced/paper/paper_metrics.json`
- `reproduced/paper/rq1_detection.csv`
- `reproduced/paper/rq2_calibration.csv`
- `reproduced/paper/reliability/*.png`
- `reproduced/rq3/security_deferral.json`
- `reproduced/rq3/*.csv`
- `reproduced/rq3/*.png`

## Reproducibility Notes

The reproduction script validates exact sample-ID alignment, duplicate IDs, label consistency, finite probabilities in `[0, 1]`, and substantive equality against the released reference JSON files. It does not refit temperatures, reviewer thresholds, judge prompts, or gate policies on confirmatory labels. The bundled RQ3 reference is compact: it keeps the primary endpoints and operating-point ablations, while the full security-deferral curves are regenerated under `reproduced/rq3/`.

See `docs/METHOD_PARAMETERS.md`, `docs/VERSIONS.txt`, and `docs/DATASETS.md` for the exact method settings and artifact boundary.

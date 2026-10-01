# Method Parameters

## Calibration Metrics

- ECE uses 10 equal-width bins over `[0, 1]`.
- Bins are `[lo, hi)` except the final bin, which includes probability `1.0`.
- NLL is computed with scikit-learn `log_loss` using natural logarithms and float64 machine-precision clipping.
- Brier score is the mean squared error between labels and probabilities.

## Temperature Scaling

- Probabilities are clipped to `[1e-7, 1 - 1e-7]` before conversion to logits.
- One scalar temperature `T` is fitted by minimizing unweighted binary NLL.
- Optimization uses SciPy bounded scalar minimization with bounds `[0.05, 20]`, `xatol=1e-5`, and `maxiter=500`.
- Confirmatory labels are never used to fit temperatures.

## Frozen Temperatures

- PG2-22M: `4.685748584103503`
- PG2-86M: `6.628761834166339`
- LLM judge: `1.590979118195362`

## Frozen Judge and Gate Parameters

- Selected judge prompt ID: `llm-judge-candidate-effect-v1`
- Selected judge prompt SHA-256: `dc15a645fcfdf3f6642175a30129f0fbd54a0271ac349a1c52c7bb23646cbee7`
- Reviewer threshold: recorded in `artifacts/calibration/judge_operating_point.json`
- Gate policy: recorded in `artifacts/calibration/gate_policy.json`
- RQ3 target FNR: `0.01`
- RQ3 costs: false positive `1.0`, false negative `100.0`, review `0.5`

## Split Sizes

- Development: 1,000 sample IDs
- Reviewer calibration: 1,000 sample IDs
- Gate calibration: 2,000 sample IDs
- Feasibility: 1,000 sample IDs
- Confirmatory: 47,611 sample IDs

# Datasets and Artifact Boundary

This artifact redistributes only sample IDs, labels, dataset/partition metadata, frozen prompts, calibration artifacts, prediction artifacts, and derived metrics. It does not redistribute raw benchmark prompt text.

## Dataset Roles

- TensorTrust: prompt-injection test set used in confirmatory evaluation.
- BIPIA: prompt-injection test set with code, email, and table partitions.
- NotInject: hard benign and attack-like benign/negative partitions used to stress false positives.
- PIDS-Bench: in-distribution, obfuscated, structural OOD, domain OOD, hard benign, and validation-derived partitions.

## Local Split Mapping

- `splits/development_ids.txt`: reviewer prompt-development split.
- `splits/reviewer_calibration_ids.txt`: reviewer probability calibration and threshold split.
- `splits/gate_calibration_ids.txt`: detector/gate calibration and gate-policy split.
- `splits/feasibility_ids.txt`: small feasibility run.
- `splits/confirmatory_ids.txt`: main confirmatory test run.

## Raw Data

To rerun model inference from raw prompts, obtain each upstream benchmark from its original source and follow its license. This release intentionally contains no raw prompt text, no private SDK configuration, and no API client code.

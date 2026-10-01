#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [ -x "${REPO_DIR}/.venv/bin/python" ]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif [ -x "${REPO_DIR}/../.venv/bin/python" ]; then
  PYTHON_BIN="${REPO_DIR}/../.venv/bin/python"
elif [ -x "${REPO_DIR}/../../.venv/bin/python" ]; then
  PYTHON_BIN="${REPO_DIR}/../../.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
OUTPUT_DIR="${REPO_DIR}/reproduced"

if [ -e "${OUTPUT_DIR}" ]; then
  echo "Refusing to overwrite existing output directory: ${OUTPUT_DIR}" >&2
  echo "Remove it before rerunning." >&2
  exit 2
fi

export PYTHONPATH="${REPO_DIR}/src"
export MPLCONFIGDIR="${OUTPUT_DIR}/.matplotlib"

echo "[1/5] validate release inputs"
"${PYTHON_BIN}" - <<'PY' "${REPO_DIR}"
import json
import math
import sys
from pathlib import Path

repo = Path(sys.argv[1])
expected_counts = {
    "development_ids.txt": 1000,
    "reviewer_calibration_ids.txt": 1000,
    "gate_calibration_ids.txt": 2000,
    "feasibility_ids.txt": 1000,
    "confirmatory_ids.txt": 47611,
}
for name, expected in expected_counts.items():
    path = repo / "splits" / name
    ids = path.read_text(encoding="utf-8").splitlines()
    if len(ids) != expected or len(ids) != len(set(ids)) or any(not item for item in ids):
        raise SystemExit(f"invalid split file {path}: expected {expected} unique non-empty IDs")
confirmatory_ids = (repo / "splits" / "confirmatory_ids.txt").read_text(encoding="utf-8").splitlines()
expected_set = set(confirmatory_ids)
prediction_names = [
    "jev_native.jsonl",
    "pg2_22m_raw.jsonl",
    "pg2_22m_calibrated.jsonl",
    "pg2_86m_raw.jsonl",
    "pg2_86m_calibrated.jsonl",
    "llm_judge_raw.jsonl",
    "llm_judge_calibrated.jsonl",
    "llm_judge_operating_point.jsonl",
]
for name in prediction_names:
    path = repo / "artifacts" / "predictions" / name
    seen = set()
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rows += 1
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise SystemExit(f"duplicate prediction ID in {path}: {sample_id}")
            seen.add(sample_id)
            probability = float(row["probability"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise SystemExit(f"invalid probability in {path}: {sample_id}")
    if rows != 47611 or seen != expected_set:
        raise SystemExit(f"prediction IDs do not match confirmatory split: {path}")
PY

echo "[2/5] regenerate RQ1/RQ2 metrics and reliability figures"
"${PYTHON_BIN}" -m evaluate_paper_results \
  --samples "${REPO_DIR}/artifacts/predictions/confirmatory_samples.jsonl" \
  --predictions "{\"jev_native\":\"${REPO_DIR}/artifacts/predictions/jev_native.jsonl\",\"llm_judge_calibrated\":\"${REPO_DIR}/artifacts/predictions/llm_judge_calibrated.jsonl\",\"llm_judge_raw\":\"${REPO_DIR}/artifacts/predictions/llm_judge_raw.jsonl\",\"pg2_22m_calibrated\":\"${REPO_DIR}/artifacts/predictions/pg2_22m_calibrated.jsonl\",\"pg2_22m_raw\":\"${REPO_DIR}/artifacts/predictions/pg2_22m_raw.jsonl\",\"pg2_86m_calibrated\":\"${REPO_DIR}/artifacts/predictions/pg2_86m_calibrated.jsonl\",\"pg2_86m_raw\":\"${REPO_DIR}/artifacts/predictions/pg2_86m_raw.jsonl\"}" \
  --output-dir "${OUTPUT_DIR}/paper"

echo "[3/5] regenerate RQ3 security-deferral curves"
"${PYTHON_BIN}" - <<'PY' "${REPO_DIR}" "${OUTPUT_DIR}/rq3"
import json
import sys
import tempfile
from pathlib import Path

from security_deferral import evaluate_security_deferral_files, write_security_deferral_artifacts

repo = Path(sys.argv[1])
output = Path(sys.argv[2])
config = json.loads((repo / "artifacts" / "calibration" / "rq3_config.json").read_text(encoding="utf-8"))
def resolve(value):
    if isinstance(value, str) and (value.startswith("artifacts/") or value.startswith("splits/")):
        return str(repo / value)
    if isinstance(value, dict):
        return {key: resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item) for item in value]
    return value
result = evaluate_security_deferral_files(**resolve(config))
write_security_deferral_artifacts(result, output, render_figure=True)
PY

echo "[4/5] compare regenerated metrics to released references"
"${PYTHON_BIN}" - <<'PY' "${REPO_DIR}" "${OUTPUT_DIR}"
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
out = Path(sys.argv[2])

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def strip_paper(value):
    value = dict(value)
    value.pop("provenance", None)
    return value

expected_paper = strip_paper(load(repo / "artifacts" / "paper_metrics.json"))
actual_paper = strip_paper(load(out / "paper" / "paper_metrics.json"))
if actual_paper != expected_paper:
    raise SystemExit("RQ1/RQ2 paper_metrics.json differs from released reference")

def compact_rq3(value):
    return {
        "target_fnr": value["target_fnr"],
        "primary_results": value["primary_results"],
        "operating_points": value["operating_points"],
        "infeasible_gate_policies": value.get("infeasible_gate_policies", {}),
    }

expected_rq3 = compact_rq3(load(repo / "artifacts" / "security_deferral.json"))
actual_rq3 = compact_rq3(load(out / "rq3" / "security_deferral.json"))
if actual_rq3 != expected_rq3:
    raise SystemExit("RQ3 compact reference differs from regenerated output")
PY

echo "[5/5] done"
echo "Outputs written to ${OUTPUT_DIR}"


from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Sample:
    id: str
    label: int
    dataset: str
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("sample id is required")
        if self.label not in (0, 1):
            raise ValueError("sample label must be 0 or 1")
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("sample dataset is required")
        if not isinstance(self.metadata, dict):
            raise ValueError("sample metadata must be an object")


@dataclass
class Prediction:
    sample_id: str
    label: int
    probability: float
    decision: int
    detector: str
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("prediction sample_id is required")
        if self.label not in (0, 1) or self.decision not in (0, 1):
            raise ValueError("prediction label and decision must be binary")
        self.probability = float(self.probability)
        self.latency_ms = float(self.latency_ms)
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be finite and in [0, 1]")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if not isinstance(self.detector, str) or not self.detector:
            raise ValueError("detector is required")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)


def read_jsonl(path: Path, kind: str) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{kind} JSONL must be non-empty and end with a newline: {path}")
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("expected object")
            rows.append(value)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            raise ValueError(f"invalid {kind} line {number}: {exc}") from exc
    return rows


def load_samples(path: Path) -> list[Sample]:
    samples, seen = [], set()
    for number, row in enumerate(read_jsonl(path, "sample"), 1):
        row = dict(row)
        row.pop("text", None)
        try:
            sample = Sample(**row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid sample line {number}: {exc}") from exc
        if sample.id in seen:
            raise ValueError(f"duplicate sample ID: {sample.id}")
        seen.add(sample.id)
        samples.append(sample)
    return samples


def load_predictions(path: Path) -> list[Prediction]:
    predictions, seen = [], set()
    for number, row in enumerate(read_jsonl(path, "prediction"), 1):
        try:
            prediction = Prediction(**row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction line {number}: {exc}") from exc
        if prediction.sample_id in seen:
            raise ValueError(f"duplicate prediction ID: {prediction.sample_id}")
        seen.add(prediction.sample_id)
        predictions.append(prediction)
    return predictions


def align_predictions(samples: list[Sample], predictions: list[Prediction], name: str) -> list[Prediction]:
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    expected = {sample.id for sample in samples}
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        extra = sorted(set(by_id) - expected)
        raise ValueError(f"{name} ID mismatch: missing={missing[:5]}, extra={extra[:5]}")
    aligned = [by_id[sample.id] for sample in samples]
    for sample, prediction in zip(samples, aligned):
        if sample.label != prediction.label:
            raise ValueError(f"{name} label mismatch for {sample.id}")
    return aligned


def atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        tmp = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        tmp.replace(path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


import argparse
import csv
from collections.abc import Mapping, Sequence
from itertools import groupby

DEFAULT_DATASET_GROUPS = {
    "tensortrust": ["tensortrust"],
    "bipia": ["bipia"],
    "notinject_hard_benign": ["notinject", "hard_benign"],
    "pids_shifts": ["pids_bench_v3"],
}
DEFAULT_PARTITION_FIELDS = ("test_partition", "category")
TABLE_COLUMNS = ("model", "scope", "status", "target_fnr", "escalation_count", "escalation_rate", "automatic_coverage", "fnr", "fpr", "normalized_cost", "latency_ms")
ABLATION_COLUMNS = ("model", "scope", "ablation", "escalation_count", "escalation_rate", "automatic_coverage", "fnr", "fpr", "normalized_cost", "latency_ms")
CURVE_COLUMNS = ("model", "scope", "escalation_count", "escalation_rate", "automatic_coverage", "tn", "fp", "fn", "tp", "fnr", "fpr", "normalized_cost", "latency_ms")


def _binary(values, name):
    result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(value not in (0, 1) for value in result):
        raise ValueError(f"{name} must be binary")
    return [int(value) for value in result]


def _finite(values, name, *, probability=False):
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or (probability and not 0 <= value <= 1) for value in result):
        raise ValueError(f"{name} must contain finite values" + (" in [0, 1]" if probability else ""))
    return result


def _confusion(labels, decisions):
    tn = sum(label == 0 and decision == 0 for label, decision in zip(labels, decisions))
    fp = sum(label == 0 and decision == 1 for label, decision in zip(labels, decisions))
    fn = sum(label == 1 and decision == 0 for label, decision in zip(labels, decisions))
    tp = sum(label == 1 and decision == 1 for label, decision in zip(labels, decisions))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def build_real_deferral_curve(labels, gate_probabilities, judge_decisions, *, target_fnr=0.01, gate_decisions=None, gate_latencies_ms=None, judge_latencies_ms=None, false_positive_cost=1.0, false_negative_cost=1.0, review_cost=0.0, include_escalated_indices=False):
    y = _binary(labels, "labels")
    probabilities = _finite(gate_probabilities, "gate_probabilities", probability=True)
    judge = _binary(judge_decisions, "judge_decisions")
    n = len(y)
    if len(probabilities) != n or len(judge) != n:
        raise ValueError("labels, gate probabilities, and judge decisions must have equal length")
    gate = _binary(gate_decisions, "gate_decisions") if gate_decisions is not None else [int(p >= 0.5) for p in probabilities]
    gate_latency = _finite([0.0] * n if gate_latencies_ms is None else gate_latencies_ms, "gate_latencies_ms")
    judge_latency = _finite([0.0] * n if judge_latencies_ms is None else judge_latencies_ms, "judge_latencies_ms")
    costs = _finite([false_positive_cost, false_negative_cost, review_cost], "costs")
    if any(cost < 0 for cost in costs):
        raise ValueError("costs must be non-negative")
    distances = [abs(probability - 0.5) for probability in probabilities]
    order = sorted(range(n), key=distances.__getitem__)
    confusion = _confusion(y, gate)
    count = 0
    latency = sum(gate_latency)
    escalated = []
    rows = []

    def append_row():
        fn, tp, fp, tn = confusion["fn"], confusion["tp"], confusion["fp"], confusion["tn"]
        total_cost = confusion["fp"] * costs[0] + confusion["fn"] * costs[1] + count * costs[2]
        row = {"escalation_count": count, "escalation_rate": count / n, "automatic_coverage": (n - count) / n, "confusion": dict(confusion), "fnr": None if fn + tp == 0 else fn / (fn + tp), "fpr": None if fp + tn == 0 else fp / (fp + tn), "normalized_cost": total_cost / n, "latency_ms": latency / n}
        if include_escalated_indices:
            row["escalated_indices"] = sorted(escalated)
        rows.append(row)

    append_row()
    cells = (("tn", "fp"), ("fn", "tp"))
    for _, indices in groupby(order, key=distances.__getitem__):
        for index in indices:
            confusion[cells[y[index]][gate[index]]] -= 1
            confusion[cells[y[index]][judge[index]]] += 1
            count += 1
            latency += judge_latency[index]
            if include_escalated_indices:
                escalated.append(index)
        append_row()
    return rows


def select_minimum_escalation(rows, *, target_fnr=0.01):
    feasible = [row for row in rows if row.get("fnr") is not None and float(row["fnr"]) <= target_fnr]
    if not feasible:
        return {"status": "infeasible", "target_fnr": float(target_fnr)}
    return dict(min(feasible, key=lambda row: (int(row["escalation_count"]), float(row["escalation_rate"]))))


def _metadata_values(value):
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {str(item) for item in values if item is not None and not isinstance(item, (dict, list, tuple, set)) and str(item)}


def _scopes(samples, dataset_groups, partition_fields):
    result = {"overall": list(range(len(samples)))}
    for dataset in sorted({sample.dataset for sample in samples}):
        result[f"dataset:{dataset}"] = [i for i, sample in enumerate(samples) if sample.dataset == dataset]
    for name, datasets in sorted(dataset_groups.items()):
        indices = [i for i, sample in enumerate(samples) if sample.dataset in set(datasets)]
        if indices:
            result[f"group:{name}"] = indices
    for dataset in sorted({sample.dataset for sample in samples}):
        for field in partition_fields:
            values = sorted({value for sample in samples if sample.dataset == dataset for value in _metadata_values(sample.metadata.get(field))})
            for value in values:
                result[f"partition:{dataset}:{field}:{value}"] = [i for i, sample in enumerate(samples) if sample.dataset == dataset and value in _metadata_values(sample.metadata.get(field))]
    return result


def _curve(samples, gate, judge, indices, target_fnr, costs):
    return build_real_deferral_curve([samples[i].label for i in indices], [gate[i].probability for i in indices], [judge[i].decision for i in indices], target_fnr=target_fnr, include_escalated_indices=False, gate_decisions=[gate[i].decision for i in indices], gate_latencies_ms=[gate[i].latency_ms for i in indices], judge_latencies_ms=[judge[i].latency_ms for i in indices], **costs)


def _select_review_indices(gate, indices, lower, upper):
    return {i for i in indices if lower <= gate[i].probability < upper}


def _operating_point_rows(model, scope, samples, gate, judge, indices, definitions, costs):
    rows = []
    for ablation, by_model in sorted(definitions.items()):
        if model not in by_model:
            continue
        config = by_model[model]
        lower, upper = float(config["allow_below"]), float(config["block_at_or_above"])
        selected = _select_review_indices(gate, indices, lower, upper)
        labels = [samples[i].label for i in indices]
        decisions = []
        for i in indices:
            if gate[i].probability < lower:
                decisions.append(0)
            elif gate[i].probability >= upper:
                decisions.append(1)
            elif config.get("review_outcome") == "oracle":
                decisions.append(samples[i].label)
            elif config.get("review_outcome") == "fail_closed":
                decisions.append(1)
            else:
                decisions.append(judge[i].decision)
        c = _confusion(labels, decisions)
        total = len(indices)
        total_cost = c["fp"] * costs["false_positive_cost"] + c["fn"] * costs["false_negative_cost"] + len(selected) * costs["review_cost"]
        rows.append({"model": model, "scope": scope, "ablation": ablation, "allow_below": lower, "block_at_or_above": upper, "escalation_count": len(selected), "escalation_rate": len(selected) / total, "automatic_coverage": (total - len(selected)) / total, "confusion": c, "fnr": None if c["fn"] + c["tp"] == 0 else c["fn"] / (c["fn"] + c["tp"]), "fpr": None if c["fp"] + c["tn"] == 0 else c["fp"] / (c["fp"] + c["tn"]), "normalized_cost": total_cost / total, "latency_ms": (sum(gate[i].latency_ms for i in indices) + sum(judge[i].latency_ms for i in selected)) / total})
    return rows


def evaluate_security_deferral_files(samples_path, judge_predictions_path, gate_prediction_paths, *, target_fnr=0.01, dataset_groups=None, partition_fields=None, operating_points=None, infeasible_gate_policies=None, false_positive_cost=1.0, false_negative_cost=1.0, review_cost=0.0):
    samples_file, judge_file = Path(samples_path), Path(judge_predictions_path)
    samples = load_samples(samples_file)
    judge = align_predictions(samples, load_predictions(judge_file), "judge")
    scopes = _scopes(samples, DEFAULT_DATASET_GROUPS if dataset_groups is None else dataset_groups, DEFAULT_PARTITION_FIELDS if partition_fields is None else partition_fields)
    costs = {"false_positive_cost": false_positive_cost, "false_negative_cost": false_negative_cost, "review_cost": review_cost}
    curves, primary, ablations, gate_provenance = {}, [], [], {}
    for model, path in sorted(gate_prediction_paths.items()):
        gate_path = Path(path)
        gate = align_predictions(samples, load_predictions(gate_path), model)
        gate_provenance[model] = {"path": str(gate_path), "sha256": sha256(gate_path)}
        curves[model] = {}
        for scope, indices in scopes.items():
            rows = _curve(samples, gate, judge, indices, target_fnr, costs)
            curves[model][scope] = rows
            endpoint = select_minimum_escalation(rows, target_fnr=target_fnr)
            primary.append({"model": model, "scope": scope, "status": "feasible", "target_fnr": target_fnr, **endpoint} if endpoint.get("status") != "infeasible" else {"model": model, "scope": scope, **endpoint})
            ablations.extend(_operating_point_rows(model, scope, samples, gate, judge, indices, operating_points or {}, costs))
    return {"target_fnr": target_fnr, "curves": curves, "primary_results": primary, "operating_points": ablations, "infeasible_gate_policies": dict(infeasible_gate_policies or {}), "provenance": {"samples": {"path": str(samples_file), "sha256": sha256(samples_file)}, "judge_predictions": {"path": str(judge_file), "sha256": sha256(judge_file)}, "gate_predictions": gate_provenance}}


def _machine(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def _write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: _machine(row.get(column)) for column in columns} for row in rows)


def _flatten_curves(curves):
    rows = []
    for model, scopes in curves.items():
        for scope, points in scopes.items():
            for point in points:
                rows.append({"model": model, "scope": scope, **point, **point.get("confusion", {})})
    return rows


def _render_figure(rows, path, target_fnr):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    series = {}
    for row in rows:
        series.setdefault((row["model"], row["scope"]), []).append(row)
    for (model, scope), points in series.items():
        axis.plot([p["escalation_rate"] for p in points], [p["fnr"] for p in points], label=f"{model} · {scope}")
    axis.axhline(target_fnr, color="black", linestyle="--", label=f"FNR target ({target_fnr * 100:g}%)")
    axis.set(xlabel="LLM escalation rate", ylabel="System FNR", xlim=(0, 1), ylim=(0, 1))
    axis.grid(True)
    if series:
        axis.legend()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_security_deferral_artifacts(result, output_dir, *, render_figure=True):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "security_deferral.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    curve_rows = _flatten_curves(result["curves"])
    _write_csv(output / "security_deferral_curves.csv", curve_rows, CURVE_COLUMNS)
    _write_csv(output / "rq3_primary.csv", result["primary_results"], TABLE_COLUMNS)
    _write_csv(output / "operating_point_ablations.csv", result["operating_points"], ABLATION_COLUMNS)
    _write_csv(output / "fnr_vs_escalation.csv", curve_rows, ("model", "scope", "escalation_count", "escalation_rate", "fnr"))
    if render_figure:
        _render_figure(curve_rows, output / "fnr_vs_escalation.png", float(result["target_fnr"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate RQ3 security-deferral artifacts from frozen predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-render-figure", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = evaluate_security_deferral_files(**config)
    write_security_deferral_artifacts(result, args.output_dir, render_figure=not args.no_render_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

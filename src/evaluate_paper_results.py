
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
from collections import defaultdict

import numpy as np

from figures import write_reliability_diagram
from metrics import compute_calibration_metrics, compute_detection_metrics, compute_false_negative_confidence


def _scopes(samples):
    scopes = {"overall": list(range(len(samples)))}
    for dataset in sorted({sample.dataset for sample in samples}):
        scopes[f"dataset:{dataset}"] = [i for i, sample in enumerate(samples) if sample.dataset == dataset]
        partitions = sorted({str(sample.metadata.get("test_partition", dataset)) for sample in samples if sample.dataset == dataset})
        for partition in partitions:
            scopes[f"partition:{dataset}:{partition}"] = [
                i for i, sample in enumerate(samples)
                if sample.dataset == dataset and str(sample.metadata.get("test_partition", dataset)) == partition
            ]
    return scopes


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _metrics(labels, probabilities, latencies):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    detection = {key: _finite(value) for key, value in compute_detection_metrics(labels, probabilities).items()}
    decisions = probabilities >= 0.5
    negatives = labels == 0
    detection["fpr"] = None if not np.any(negatives) else float(np.mean(decisions[negatives]))
    calibration = {key: _finite(value) for key, value in compute_calibration_metrics(labels, probabilities).items()}
    misses = compute_false_negative_confidence(labels, probabilities)
    false_negative_confidence = {"count": misses["count"], "mean": _finite(misses["mean"]), "max": _finite(misses["max"])}
    latency = np.asarray(latencies, dtype=float)
    return {
        "sample_count": len(labels),
        "class_support": {"0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))},
        "detection": detection,
        "calibration": calibration,
        "false_negative_confidence": false_negative_confidence,
        "latency_ms": {"mean": float(np.mean(latency)), "p50": float(np.percentile(latency, 50)), "p95": float(np.percentile(latency, 95))},
    }


def _write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_paper_results(samples_path, prediction_paths, output_dir):
    samples_file, output = Path(samples_path), Path(output_dir)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    samples = load_samples(samples_file)
    sample_ids = {sample.id for sample in samples}
    scopes = _scopes(samples)
    models = {}
    provenance = {"samples": {"path": str(samples_file), "sha256": sha256(samples_file)}, "predictions": {}}
    aligned_by_model = {}
    for model, raw_path in sorted(prediction_paths.items()):
        path = Path(raw_path)
        aligned = align_predictions(samples, load_predictions(path), model)
        aligned_by_model[model] = aligned
        provenance["predictions"][model] = {"path": str(path), "sha256": sha256(path)}
    output.mkdir(parents=True)
    reliability = output / "reliability"
    reliability.mkdir()
    rq1_rows, rq2_rows = [], []
    for model, aligned in aligned_by_model.items():
        models[model] = {}
        for scope, indices in scopes.items():
            labels = [samples[i].label for i in indices]
            probabilities = [aligned[i].probability for i in indices]
            metrics = _metrics(labels, probabilities, [aligned[i].latency_ms for i in indices])
            models[model][scope] = metrics
            rq1_rows.append({"model": model, "scope": scope, **metrics["class_support"], **metrics["detection"]})
            rq2_rows.append({"model": model, "scope": scope, **metrics["class_support"], **metrics["calibration"], **{f"false_negative_{key}": value for key, value in metrics["false_negative_confidence"].items()}})
            safe_scope = scope.replace(":", "_")
            write_reliability_diagram(np.asarray(labels), np.asarray(probabilities), reliability / f"{model}.{safe_scope}.png")
    result = {"schema_version": 1, "models": models, "provenance": provenance}
    (output / "paper_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output / "rq1_detection.csv", rq1_rows, ("model", "scope", "0", "1", "auroc", "auprc", "tpr_at_1pct_fpr", "fnr", "fpr"))
    _write_csv(output / "rq2_calibration.csv", rq2_rows, ("model", "scope", "0", "1", "ece", "brier", "nll", "false_negative_count", "false_negative_mean", "false_negative_max"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate RQ1/RQ2 paper metrics from frozen predictions.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--predictions", required=True, help="JSON object mapping model names to prediction JSONL paths")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    evaluate_paper_results(args.samples, json.loads(args.predictions), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

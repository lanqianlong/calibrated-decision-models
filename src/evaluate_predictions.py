
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

import numpy as np

from apply_temperature import calibrate_probability
from figures import write_reliability_diagram
from metrics import compute_calibration_metrics, compute_detection_metrics, compute_false_negative_confidence
from temperature import fit_temperature


def _logit(prediction: Prediction):
    for key in ("raw_logit", "logit"):
        if key in prediction.metadata:
            value = prediction.metadata[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} for {prediction.sample_id} must be finite")
            return float(value), "raw_logit"
    if "logits" in prediction.metadata:
        values = prediction.metadata["logits"]
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"logits for {prediction.sample_id} must contain [benign, attack]")
        return float(values[1] - values[0]), "raw_logit"
    probability = min(max(float(prediction.probability), 1e-7), 1 - 1e-7)
    return math.log(probability / (1 - probability)), "probability"


def _write_reliability_csv(path, labels, probabilities, bins):
    edges = np.linspace(0.0, 1.0, bins + 1)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("bin_lower", "bin_upper", "count", "mean_probability", "positive_rate"), lineterminator="\n")
        writer.writeheader()
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (probabilities >= lower) & (probabilities <= upper if upper == 1 else probabilities < upper)
            writer.writerow({"bin_lower": float(lower), "bin_upper": float(upper), "count": int(np.sum(mask)), "mean_probability": float(np.mean(probabilities[mask])) if np.any(mask) else "", "positive_rate": float(np.mean(labels[mask])) if np.any(mask) else ""})


def _evaluate(labels, probabilities):
    support = {"0": int(np.sum(labels == 0)), "1": int(np.sum(labels == 1))}
    if support["0"] and support["1"]:
        detection = compute_detection_metrics(labels, probabilities)
    else:
        detection = {"auroc": None, "auprc": None, "tpr_at_1pct_fpr": None, "fnr": None if not support["1"] else float(np.mean(probabilities[labels == 1] < 0.5))}
    calibration = compute_calibration_metrics(labels, probabilities)
    misses = compute_false_negative_confidence(labels, probabilities)
    return {"detection": detection, "calibration": calibration, "false_negative_confidence": misses, "metadata": {"class_support": support}}


def evaluate_prediction_files(calibration_samples_path, calibration_predictions_path, test_sets, output_dir, *, bins=10):
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    calibration_samples = load_samples(Path(calibration_samples_path))
    calibration_predictions = align_predictions(calibration_samples, load_predictions(Path(calibration_predictions_path)), "calibration")
    labels = np.asarray([sample.label for sample in calibration_samples], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("calibration set must contain both classes")
    logits = np.asarray([_logit(prediction)[0] for prediction in calibration_predictions], dtype=float)
    artifact = fit_temperature(labels, logits, [sample.id for sample in calibration_samples])
    output.mkdir(parents=True)
    reliability = output / "reliability"
    reliability.mkdir()
    results = {"schema_version": 1, "calibration": {"temperature": artifact.temperature, "sample_count": len(labels), "samples_sha256": sha256(Path(calibration_samples_path)), "predictions_sha256": sha256(Path(calibration_predictions_path))}, "test_sets": {}}
    for name, paths in sorted(test_sets.items()):
        sample_path, prediction_path = map(Path, paths)
        samples = load_samples(sample_path)
        predictions = align_predictions(samples, load_predictions(prediction_path), name)
        y = np.asarray([sample.label for sample in samples], dtype=int)
        raw = np.asarray([prediction.probability for prediction in predictions], dtype=float)
        test_logits = np.asarray([_logit(prediction)[0] for prediction in predictions], dtype=float)
        calibrated = artifact.transform(test_logits)
        results["test_sets"][name] = {"sample_count": len(samples), "raw": _evaluate(y, raw), "calibrated": _evaluate(y, calibrated), "provenance": {"samples_sha256": sha256(sample_path), "predictions_sha256": sha256(prediction_path)}}
        _write_reliability_csv(reliability / f"{name}.raw.csv", y, raw, bins)
        _write_reliability_csv(reliability / f"{name}.calibrated.csv", y, calibrated, bins)
        write_reliability_diagram(y, raw, reliability / f"{name}.raw.png")
        write_reliability_diagram(y, calibrated, reliability / f"{name}.calibrated.png")
    (output / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit detector temperature and evaluate named prediction files.")
    parser.add_argument("--calibration-samples", required=True)
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--test-sets", required=True, help="JSON object mapping names to [samples, predictions]")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    evaluate_prediction_files(args.calibration_samples, args.calibration_predictions, json.loads(args.test_sets), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

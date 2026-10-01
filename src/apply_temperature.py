
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


def _temperature(value: dict) -> float:
    raw = value.get("temperature")
    if raw is None and isinstance(value.get("calibration"), dict):
        raw = value["calibration"].get("temperature")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError("calibration artifact requires a numeric temperature")
    result = float(raw)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("temperature must be finite and positive")
    return result


def calibrate_probability(probability: float, temperature: float) -> float:
    clipped = min(max(float(probability), 1e-7), 1.0 - 1e-7)
    logit = math.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def apply_temperature_file(predictions_path, artifact_path, output_path, manifest_path):
    predictions_file, artifact_file = Path(predictions_path), Path(artifact_path)
    output, manifest = Path(output_path), Path(manifest_path)
    artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
    temperature = _temperature(artifact)
    rows = []
    for prediction in load_predictions(predictions_file):
        calibrated = calibrate_probability(prediction.probability, temperature)
        metadata = dict(prediction.metadata)
        metadata.update({"raw_probability": prediction.probability, "temperature": temperature, "calibration_sha256": sha256(artifact_file)})
        transformed = Prediction(prediction.sample_id, prediction.label, calibrated, int(calibrated >= 0.5), f"{prediction.detector}:temperature-scaled", prediction.latency_ms, metadata)
        rows.append((transformed.to_json() + "\n").encode())
    payload = b"".join(rows)
    result = {"schema_version": 1, "record_count": len(rows), "temperature": temperature, "input_predictions": {"path": str(predictions_file), "sha256": sha256(predictions_file)}, "calibration_artifact": {"path": str(artifact_file), "sha256": sha256(artifact_file)}, "output": {"path": str(output), "sha256": hashlib.sha256(payload).hexdigest()}}
    atomic_write(output, payload)
    atomic_write(manifest, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply a frozen scalar temperature to Prediction JSONL.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    apply_temperature_file(args.predictions, args.artifact, args.output, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

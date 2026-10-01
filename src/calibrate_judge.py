
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

import numpy as np

from metrics import compute_calibration_metrics
from temperature import fit_temperature


def _logits(probabilities):
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def _replay_value(temperature: float, predictions):
    probabilities = 1 / (1 + np.exp(-_logits([p.probability for p in predictions]) / temperature))
    encoded = json.dumps(probabilities.tolist(), separators=(",", ":")).encode()
    return {"count": len(probabilities), "calibrated_probabilities_sha256": hashlib.sha256(encoded).hexdigest()}


def calibrate_judge(samples_path, predictions_path, output_path):
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"calibration artifact already exists: {output_path}")
    samples = load_samples(Path(samples_path))
    predictions = align_predictions(samples, load_predictions(Path(predictions_path)), "judge_calibration")
    labels = np.asarray([sample.label for sample in samples], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("judge calibration requires both classes")
    prompt_hashes = {prediction.metadata.get("prompt_hash") for prediction in predictions}
    if len(prompt_hashes) != 1 or None in prompt_hashes:
        raise ValueError("predictions require one immutable prompt_hash")
    probabilities = np.asarray([prediction.probability for prediction in predictions])
    logits = _logits(probabilities)
    artifact = fit_temperature(labels, logits, [sample.id for sample in samples])
    calibrated = artifact.transform(logits)
    result = {
        "format_version": 1,
        "temperature": artifact.temperature,
        "sample_ids": artifact.sample_ids,
        "metrics": {"pre": compute_calibration_metrics(labels, probabilities), "post": compute_calibration_metrics(labels, calibrated)},
        "replay": _replay_value(artifact.temperature, predictions),
        "provenance": {"samples_sha256": sha256(Path(samples_path)), "predictions_sha256": sha256(Path(predictions_path)), "prompt_hash": prompt_hashes.pop(), "model_ids": sorted({p.detector for p in predictions})},
    }
    atomic_write(output_path, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return result


def replay_judge_calibration(artifact_path, predictions_path):
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if sha256(Path(predictions_path)) != artifact["provenance"]["predictions_sha256"]:
        raise ValueError("prediction checksum does not match calibration provenance")
    predictions = load_predictions(Path(predictions_path))
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    ordered = [by_id[sample_id] for sample_id in artifact["sample_ids"]]
    replay = _replay_value(float(artifact["temperature"]), ordered)
    if replay != artifact["replay"]:
        raise ValueError("calibrated probability replay checksum does not match")
    return replay


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit LLM judge temperature on a calibration split.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    calibrate_judge(args.samples, args.predictions, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

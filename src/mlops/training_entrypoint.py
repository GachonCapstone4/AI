from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import SBERT_BASE_MODEL
from data_utils import generate_contrastive_pairs, load_dataset
from train_domain import train_domain_classifier
from train_intent import train_intent_classifiers
from train_sbert import generate_embeddings, run_sbert_finetuning


ARTIFACT_FORMAT_VERSION = "1.0"
MODEL_TYPE = "sbert_logistic_regression"
METRIC_SOURCE = "training_set"
METRIC_WARNING = "Metrics are computed on the training set and should be replaced with validation metrics."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "sbert_dir": output_dir / "sbert",
        "domain_model": output_dir / "domain_model.pkl",
        "intent_model": output_dir / "intent_model.pkl",
        "metrics": output_dir / "metrics.json",
        "config": output_dir / "config.json",
        "label_mapping": output_dir / "label_mapping.json",
    }


def _build_label_mapping(le_domain, intent_encoders: dict) -> dict:
    return {
        "domains": le_domain.classes_.tolist(),
        "intents": {
            str(domain): encoder.classes_.tolist()
            for domain, encoder in intent_encoders.items()
        },
    }


def _compute_metrics(df, X, domain_clf, le_domain, intent_classifiers, intent_encoders) -> dict:
    domain_pred_enc = domain_clf.predict(X)
    domain_pred = le_domain.inverse_transform(domain_pred_enc)
    domain_accuracy = accuracy_score(df["domain"].values, domain_pred)

    intent_true: list[str] = []
    intent_pred: list[str] = []
    for domain, clf in intent_classifiers.items():
        mask = df["domain"] == domain
        if not np.any(mask):
            continue

        encoder = intent_encoders[domain]
        y_pred_enc = clf.predict(X[mask])
        y_pred = encoder.inverse_transform(y_pred_enc)
        intent_true.extend(df.loc[mask, "intent"].tolist())
        intent_pred.extend(y_pred.tolist())

    # TODO: replace this training-set metric with a held-out or cross-validation
    # metric when the MLOps dataset manifest includes a train/validation split.
    intent_f1 = (
        f1_score(intent_true, intent_pred, average="weighted")
        if intent_true and intent_pred else 0.0
    )

    return {
        "domain_accuracy": round(float(domain_accuracy), 4),
        "intent_f1": round(float(intent_f1), 4),
    }


def run_training(dataset_path: Path, output_dir: Path, model_version: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = _artifact_paths(output_dir)
    sbert_dir = artifact_paths["sbert_dir"]
    domain_model_path = artifact_paths["domain_model"]
    intent_model_path = artifact_paths["intent_model"]

    df = load_dataset(str(dataset_path))
    pairs = generate_contrastive_pairs(df)

    run_sbert_finetuning(
        output_path=str(sbert_dir),
        base_model=SBERT_BASE_MODEL,
        pairs=pairs,
    )
    X = generate_embeddings(
        df["email_text"].tolist(),
        model_path=str(sbert_dir),
        save_path=None,
    )

    domain_clf, le_domain = train_domain_classifier(
        X,
        df["domain"].values,
        model_path=domain_model_path,
        label_encoder_path=None,
    )
    intent_classifiers, intent_encoders = train_intent_classifiers(
        X,
        df,
        model_path=intent_model_path,
        label_encoders_path=None,
    )

    joblib.dump(
        {
            "classifier": domain_clf,
            "label_encoder": le_domain,
        },
        domain_model_path,
    )
    joblib.dump(
        {
            "classifiers": intent_classifiers,
            "label_encoders": intent_encoders,
        },
        intent_model_path,
    )

    created_at = _utc_now()
    metric_values = _compute_metrics(
        df,
        X,
        domain_clf,
        le_domain,
        intent_classifiers,
        intent_encoders,
    )

    metrics = {
        "model_version": model_version,
        **metric_values,
        "created_at": created_at,
        "metric_source": METRIC_SOURCE,
        "warning": METRIC_WARNING,
    }
    config = {
        "model_version": model_version,
        "base_model": SBERT_BASE_MODEL,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_type": MODEL_TYPE,
    }
    label_mapping = _build_label_mapping(le_domain, intent_encoders)

    _write_json(artifact_paths["metrics"], metrics)
    _write_json(artifact_paths["config"], config)
    _write_json(artifact_paths["label_mapping"], label_mapping)

    return {
        "output_dir": str(output_dir),
        "model_version": model_version,
        "metrics": metrics,
    }


def build_dry_run_plan(dataset_path: Path, output_dir: Path, model_version: str) -> dict:
    artifact_paths = _artifact_paths(output_dir)
    return {
        "dry_run": True,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "model_version": model_version,
        "will_train": [
            "sbert",
            "domain_logistic_regression",
            "intent_logistic_regression",
        ],
        "artifact_files": {
            "sbert": str(artifact_paths["sbert_dir"]),
            "domain_model": str(artifact_paths["domain_model"]),
            "intent_model": str(artifact_paths["intent_model"]),
            "metrics": str(artifact_paths["metrics"]),
            "config": str(artifact_paths["config"]),
            "label_mapping": str(artifact_paths["label_mapping"]),
        },
        "pickle_structure": {
            "domain_model.pkl": {
                "classifier": "sklearn LogisticRegression",
                "label_encoder": "sklearn LabelEncoder",
            },
            "intent_model.pkl": {
                "classifiers": "dict[str, sklearn LogisticRegression]",
                "label_encoders": "dict[str, sklearn LabelEncoder]",
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SBERT + Logistic Regression models and write a standard model artifact."
    )
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned artifact structure without reading data or training models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        result = build_dry_run_plan(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            model_version=args.model_version,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    result = run_training(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_version=args.model_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.mlops.s3_client import plan_directory_upload, validate_model_artifact_dir
from src.mlops.training_container_entrypoint import (
    build_dry_run_plan,
    build_latest_pointer_key,
    build_model_artifact_prefix,
    parse_args,
)


def _write_standard_artifact(root: Path) -> None:
    sbert_dir = root / "sbert"
    sbert_dir.mkdir(parents=True)
    (sbert_dir / "model.safetensors").write_bytes(b"model")
    (sbert_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "domain_model.pkl").write_bytes(b"domain")
    (root / "intent_model.pkl").write_bytes(b"intent")
    (root / "label_mapping.json").write_text("{}", encoding="utf-8")
    (root / "metrics.json").write_text("{}", encoding="utf-8")
    (root / "config.json").write_text("{}", encoding="utf-8")


def test_model_artifact_prefix_uses_model_version_not_training_job_name() -> None:
    assert build_model_artifact_prefix("models", "training-final-004") == "models/training-final-004"
    assert build_model_artifact_prefix("/models/", "/training-final-004/") == "models/training-final-004"


def test_model_artifact_prefix_does_not_duplicate_model_version() -> None:
    prefix = build_model_artifact_prefix("models/training-final-004", "training-final-004")

    assert prefix == "models/training-final-004"
    assert prefix != "models/training-final-004/training-final-004"


def test_latest_pointer_key_uses_model_prefix_root() -> None:
    assert build_latest_pointer_key("models") == "models/latest.json"


def test_container_dry_run_targets_standard_model_prefix(tmp_path: Path) -> None:
    args = argparse.Namespace(
        job_id="sagemaker-job-20260511",
        training_job_name="sagemaker-job-20260511",
        dataset_path=tmp_path / "dataset.csv",
        dataset_s3_uri=None,
        model_version="training-final-004",
        output_dir=Path("/opt/ml/model"),
        s3_bucket="capstone-gachon",
        s3_model_prefix="models",
        aws_region="ap-northeast-2",
    )

    plan = build_dry_run_plan(args)

    assert plan["training_job_name"] == "sagemaker-job-20260511"
    assert plan["model_version"] == "training-final-004"
    assert plan["upload"]["prefix"] == "models/training-final-004"
    assert plan["upload"]["s3_uri"] == "s3://capstone-gachon/models/training-final-004/"
    assert plan["latest_pointer"]["s3_uri"] == "s3://capstone-gachon/models/latest.json"


def test_parse_args_supports_required_s3_environment_aliases(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["training_container_entrypoint.py", "--dry-run"])
    monkeypatch.setenv("JOB_ID", "status-job-1")
    monkeypatch.setenv("TRAINING_JOB_NAME", "sagemaker-job-1")
    monkeypatch.setenv("MODEL_VERSION", "training-final-004")
    monkeypatch.setenv("DATASET_PATH", "dataset.csv")
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setenv("S3_MODEL_BUCKET", "capstone-gachon")
    monkeypatch.delenv("S3_MODEL_PREFIX", raising=False)
    monkeypatch.setenv("MODEL_S3_PREFIX", "models")
    monkeypatch.delenv("AWS_REGION", raising=False)

    args = parse_args()

    assert args.job_id == "status-job-1"
    assert args.training_job_name == "sagemaker-job-1"
    assert args.model_version == "training-final-004"
    assert args.s3_bucket == "capstone-gachon"
    assert args.s3_model_prefix == "models"
    assert args.aws_region == "ap-northeast-2"


def test_upload_plan_does_not_include_local_version_directory_in_s3_key(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "training-final-004"
    _write_standard_artifact(artifact_dir)

    upload_plan = plan_directory_upload(
        artifact_dir,
        "capstone-gachon",
        "models/training-final-004",
    )
    keys = {item["key"] for item in upload_plan["files"]}

    assert "models/training-final-004/domain_model.pkl" in keys
    assert "models/training-final-004/sbert/model.safetensors" in keys
    assert "models/training-final-004/training-final-004/domain_model.pkl" not in keys
    assert not any("/output/model.tar.gz" in key for key in keys)


def test_validate_model_artifact_dir_requires_sbert_core_files(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_standard_artifact(artifact_dir)
    (artifact_dir / "sbert" / "tokenizer.json").unlink()

    with pytest.raises(FileNotFoundError, match="sbert/tokenizer.json"):
        validate_model_artifact_dir(artifact_dir)

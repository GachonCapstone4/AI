from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from .s3_client import upload_directory_to_s3, upload_json_to_s3, validate_model_artifact_dir
except ImportError:
    from src.mlops.s3_client import upload_directory_to_s3, upload_json_to_s3, validate_model_artifact_dir

from src.mlops.training_events import publish_sse_log, publish_training_status


DEFAULT_DOWNLOADED_DATASET_DIR = Path("/opt/ml/input/data")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/model")
LATEST_POINTER_KEY = "models/latest.json"


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _env_path(name: str) -> Path | None:
    value = _env(name)
    return Path(value) if value else None


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _join_s3_prefix(prefix: str, model_version: str) -> str:
    normalized_prefix = prefix.strip("/")
    normalized_version = model_version.strip("/")
    if not normalized_prefix:
        return normalized_version
    return f"{normalized_prefix}/{normalized_version}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_latest_pointer_payload(
    *,
    model_version: str,
    job_id: str,
    artifact_s3_uri: str,
    metrics: dict,
) -> dict:
    return {
        "model_version": model_version,
        "updated_at": _utc_now(),
        "job_id": job_id,
        "artifact_s3_uri": artifact_s3_uri,
        "metrics": {
            "domain_accuracy": metrics.get("domain_accuracy"),
            "intent_f1": metrics.get("intent_f1"),
        },
    }


def _resolve_dataset_path(dataset_path: Path | None, dataset_s3_uri: str | None) -> Path:
    if dataset_path is not None:
        return dataset_path
    if dataset_s3_uri is None:
        raise ValueError("Either --dataset-path or --dataset-s3-uri is required.")

    _, key = _parse_s3_uri(dataset_s3_uri)
    filename = Path(key).name
    if not filename:
        raise ValueError(f"Dataset S3 URI must point to a file: {dataset_s3_uri}")
    return DEFAULT_DOWNLOADED_DATASET_DIR / filename


def _download_dataset_from_s3(s3_uri: str, dataset_path: Path) -> dict:
    bucket, key = _parse_s3_uri(s3_uri)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    import boto3

    s3_client = boto3.client("s3")
    s3_client.download_file(bucket, key, str(dataset_path))
    return {
        "source": s3_uri,
        "bucket": bucket,
        "key": key,
        "local_path": str(dataset_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model training inside a SageMaker Training container and upload artifacts to S3."
    )
    parser.add_argument("--job-id", default=_env("JOB_ID"))
    parser.add_argument("--dataset-path", type=Path, default=_env_path("DATASET_PATH"))
    parser.add_argument("--dataset-s3-uri", default=_env("DATASET_S3_URI"))
    parser.add_argument("--model-version", default=_env("MODEL_VERSION"))
    parser.add_argument("--output-dir", type=Path, default=_env_path("OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)
    parser.add_argument("--s3-bucket", default=_env("S3_BUCKET"))
    parser.add_argument("--s3-model-prefix", default=_env("S3_MODEL_PREFIX"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without downloading data, training, validating, or uploading.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] Ignored unknown args from SageMaker: {unknown}", flush=True)
    return args


def _validate_required_args(args: argparse.Namespace) -> None:
    missing = []
    for attr, label in (
        ("job_id", "job_id"),
        ("model_version", "model_version"),
        ("output_dir", "output_dir"),
        ("s3_bucket", "s3_bucket"),
        ("s3_model_prefix", "s3_model_prefix"),
    ):
        if getattr(args, attr) in (None, ""):
            missing.append(label)

    if args.dataset_path is None and not args.dataset_s3_uri:
        missing.append("dataset_path or dataset_s3_uri")

    if missing:
        raise ValueError(f"Missing required values: {', '.join(missing)}")


def build_dry_run_plan(args: argparse.Namespace) -> dict:
    _validate_required_args(args)
    dataset_path = _resolve_dataset_path(args.dataset_path, args.dataset_s3_uri)
    artifact_prefix = _join_s3_prefix(args.s3_model_prefix, args.model_version)
    artifact_s3_uri = f"s3://{args.s3_bucket}/{artifact_prefix}/"

    return {
        "dry_run": True,
        "job_id": args.job_id,
        "dataset": {
            "s3_uri": args.dataset_s3_uri,
            "local_path": str(dataset_path),
            "will_download": bool(args.dataset_s3_uri),
        },
        "training": {
            "will_run": True,
            "entrypoint": "src.mlops.training_entrypoint.run_training",
            "dataset_path": str(dataset_path),
            "output_dir": str(args.output_dir),
            "model_version": args.model_version,
        },
        "validation": {
            "will_validate": True,
            "artifact_dir": str(args.output_dir),
        },
        "upload": {
            "will_upload": True,
            "bucket": args.s3_bucket,
            "prefix": artifact_prefix,
            "s3_uri": artifact_s3_uri,
        },
        "latest_pointer": {
            "will_update": True,
            "s3_uri": f"s3://{args.s3_bucket}/{LATEST_POINTER_KEY}",
        },
    }


def run_container_training(args: argparse.Namespace) -> dict:
    _validate_required_args(args)
    dataset_path = _resolve_dataset_path(args.dataset_path, args.dataset_s3_uri)
    artifact_prefix = _join_s3_prefix(args.s3_model_prefix, args.model_version)
    artifact_s3_uri = f"s3://{args.s3_bucket}/{artifact_prefix}/"

    try:
        publish_training_status(args.job_id, "running")

        dataset_download = None
        if args.dataset_s3_uri:
            publish_sse_log("[INFO] dataset 다운로드 시작")
            dataset_download = _download_dataset_from_s3(args.dataset_s3_uri, dataset_path)
        else:
            publish_sse_log("[INFO] 로컬 dataset 사용")

        publish_sse_log("[INFO] SBERT 학습 시작")

        from .training_entrypoint import run_training

        training_result = run_training(
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            model_version=args.model_version,
        )

        publish_sse_log("[INFO] classifier 학습 완료")
        validation = validate_model_artifact_dir(args.output_dir)

        publish_sse_log("[INFO] 모델 업로드")
        upload_result = upload_directory_to_s3(
            local_dir=args.output_dir,
            bucket=args.s3_bucket,
            prefix=artifact_prefix,
        )

        publish_sse_log("[INFO] latest.json 갱신")
        latest_pointer_payload = _build_latest_pointer_payload(
            model_version=args.model_version,
            job_id=args.job_id,
            artifact_s3_uri=artifact_s3_uri,
            metrics=training_result["metrics"],
        )
        latest_pointer_upload = upload_json_to_s3(
            payload=latest_pointer_payload,
            bucket=args.s3_bucket,
            key=LATEST_POINTER_KEY,
        )

        publish_sse_log("[INFO] 학습 완료")
        publish_training_status(
            args.job_id,
            "completed",
            model_version=args.model_version,
            metrics=training_result["metrics"],
        )

        return {
            "dry_run": False,
            "job_id": args.job_id,
            "dataset_download": dataset_download,
            "training_result": training_result,
            "validation": validation,
            "upload_result": upload_result,
            "latest_pointer": {
                "payload": latest_pointer_payload,
                "upload_result": latest_pointer_upload,
            },
            "s3_artifact_uri": artifact_s3_uri,
        }
    except Exception as exc:
        publish_training_status(args.job_id, "failed", error_message=str(exc))
        publish_sse_log(f"[ERROR] {exc}")
        raise


def main() -> None:
    args = parse_args()
    result = build_dry_run_plan(args) if args.dry_run else run_container_training(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

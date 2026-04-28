from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_VOLUME_SIZE_GB = 30
DEFAULT_MAX_RUNTIME_SECONDS = 24 * 60 * 60
DEFAULT_OUTPUT_DIR = "/opt/ml/model"
DEFAULT_REGION = "ap-northeast-2"
DEFAULT_MODEL_PREFIX = "models"


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _job_name(job_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9-]", "-", job_id).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    if not normalized:
        normalized = "training-job"
    if not normalized[0].isalpha():
        normalized = f"job-{normalized}"
    return normalized[:63].rstrip("-")


def _default_model_version(job_id: str) -> str:
    return f"{_job_name(job_id)}-{_utc_timestamp()}"[:63].rstrip("-")


def _join_s3_uri(bucket: str, *parts: str) -> str:
    key = "/".join(part.strip("/") for part in parts if part and part.strip("/"))
    return f"s3://{bucket}/{key}" if key else f"s3://{bucket}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch backend-managed SageMaker or Kubernetes jobs."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-type", required=True, choices=["training", "k8s_job"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-json")
    parser.add_argument("--manifest-path")
    parser.add_argument("--role-arn", default=_env("SAGEMAKER_ROLE_ARN"))
    parser.add_argument("--training-image-uri", default=_env("TRAINING_IMAGE_URI"))
    parser.add_argument("--s3-bucket", default=_env("S3_BUCKET"))
    parser.add_argument("--dataset-s3-uri", default=_env("S3_DATASET_URI"))
    parser.add_argument("--model-version", default=_env("MODEL_VERSION"))
    parser.add_argument("--s3-model-prefix", default=_env("S3_MODEL_PREFIX") or DEFAULT_MODEL_PREFIX)
    parser.add_argument("--aws-region", default=_env("AWS_REGION") or DEFAULT_REGION)
    parser.add_argument("--instance-type", default=_env("SAGEMAKER_INSTANCE_TYPE") or DEFAULT_INSTANCE_TYPE)
    parser.add_argument(
        "--instance-count",
        type=int,
        default=int(_env("SAGEMAKER_INSTANCE_COUNT") or DEFAULT_INSTANCE_COUNT),
    )
    parser.add_argument(
        "--volume-size-gb",
        type=int,
        default=int(_env("SAGEMAKER_VOLUME_SIZE_GB") or DEFAULT_VOLUME_SIZE_GB),
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=int(_env("SAGEMAKER_MAX_RUNTIME_SECONDS") or DEFAULT_MAX_RUNTIME_SECONDS),
    )
    return parser.parse_args()


def load_k8s_job_manifest(args: argparse.Namespace) -> dict:
    if args.manifest_json and args.manifest_path:
        raise ValueError("Use only one of --manifest-json or --manifest-path.")
    if not args.manifest_json and not args.manifest_path:
        raise ValueError("Kubernetes Job execution requires --manifest-json or --manifest-path.")

    if args.manifest_json:
        manifest = json.loads(args.manifest_json)
    else:
        manifest_path = Path(args.manifest_path)
        with manifest_path.open("r", encoding="utf-8") as file:
            if manifest_path.suffix.lower() == ".json":
                manifest = json.load(file)
            else:
                import yaml

                manifest = yaml.safe_load(file)

    if not isinstance(manifest, dict):
        raise ValueError("Kubernetes Job manifest must resolve to a JSON/YAML object.")
    return manifest


def _k8s_job_name(manifest: dict) -> str | None:
    metadata = manifest.get("metadata") or {}
    return metadata.get("name")


def inject_k8s_job_id(manifest: dict, job_id: str) -> None:
    containers = manifest["spec"]["template"]["spec"]["containers"]
    for container in containers:
        env = container.setdefault("env", [])
        if not isinstance(env, list):
            raise ValueError("Kubernetes Job container env must be a list when provided.")

        for item in env:
            if isinstance(item, dict) and item.get("name") == "JOB_ID":
                item["value"] = job_id
                break
        else:
            env.append({"name": "JOB_ID", "value": job_id})


def build_k8s_job_dry_run_output(manifest: dict) -> dict:
    from src.mlops.k8s_job_executor import (
        get_k8s_job_name,
        get_k8s_job_namespace,
        validate_k8s_job_manifest,
    )

    validate_k8s_job_manifest(manifest)

    return {
        "job_type": "k8s_job",
        "dry_run": True,
        "k8s_job_name": get_k8s_job_name(manifest),
        "namespace": get_k8s_job_namespace(manifest),
        "manifest": manifest,
    }


def _missing_required(args: argparse.Namespace) -> list[str]:
    required = (
        ("role_arn", "SAGEMAKER_ROLE_ARN or --role-arn"),
        ("training_image_uri", "TRAINING_IMAGE_URI or --training-image-uri"),
        ("s3_bucket", "S3_BUCKET or --s3-bucket"),
        ("dataset_s3_uri", "S3_DATASET_URI or --dataset-s3-uri"),
        ("model_version", "MODEL_VERSION or --model-version"),
    )
    return [label for attr, label in required if not getattr(args, attr)]


def build_training_job_config(args: argparse.Namespace, allow_placeholders: bool = False) -> dict:
    model_version = args.model_version or (
        _default_model_version(args.job_id) if allow_placeholders else None
    )
    bucket = args.s3_bucket
    region = args.aws_region

    if allow_placeholders:
        bucket = bucket or "capstone-gachon"
        role_arn = args.role_arn or "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        image_uri = (
            args.training_image_uri
            or f"123456789012.dkr.ecr.{region}.amazonaws.com/capstone-ai2-training:latest"
        )
        dataset_s3_uri = args.dataset_s3_uri or _join_s3_uri(bucket, "datasets", "dataset_new.csv")
    else:
        role_arn = args.role_arn
        image_uri = args.training_image_uri
        dataset_s3_uri = args.dataset_s3_uri

    training_job_name = _job_name(args.job_id)
    artifact_s3_uri = _join_s3_uri(bucket, args.s3_model_prefix, model_version)
    sagemaker_output_s3_uri = _join_s3_uri(
        bucket,
        args.s3_model_prefix,
        model_version,
        "sagemaker-output",
    )

    return {
        "TrainingJobName": training_job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        "OutputDataConfig": {
            "S3OutputPath": sagemaker_output_s3_uri,
        },
        "ResourceConfig": {
            "InstanceType": args.instance_type,
            "InstanceCount": args.instance_count,
            "VolumeSizeInGB": args.volume_size_gb,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": args.max_runtime_seconds,
        },
        "Environment": {
            "JOB_ID": args.job_id,
            "DATASET_S3_URI": dataset_s3_uri,
            "MODEL_VERSION": model_version,
            "OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
            "S3_BUCKET": bucket,
            "S3_MODEL_PREFIX": args.s3_model_prefix,
        },
        "Tags": [
            {"Key": "job_id", "Value": args.job_id},
            {"Key": "job_type", "Value": args.job_type},
            {"Key": "model_version", "Value": model_version},
        ],
        "_metadata": {
            "aws_region": region,
            "artifact_s3_uri": f"{artifact_s3_uri}/",
            "sagemaker_output_s3_uri": f"{sagemaker_output_s3_uri}/",
        },
    }


def _create_training_job(config: dict, region: str) -> dict:
    import boto3

    request = {key: value for key, value in config.items() if not key.startswith("_")}
    client = boto3.client("sagemaker", region_name=region)
    response = client.create_training_job(**request)
    return {
        "training_job_name": config["TrainingJobName"],
        "training_job_arn": response.get("TrainingJobArn"),
        "sagemaker_response": response,
        "artifact_s3_uri": config["_metadata"]["artifact_s3_uri"],
    }


def main() -> None:
    args = parse_args()

    if args.job_type == "k8s_job":
        manifest = load_k8s_job_manifest(args)
        from src.mlops.k8s_job_executor import validate_k8s_job_manifest

        validate_k8s_job_manifest(manifest)
        inject_k8s_job_id(manifest, args.job_id)
        if args.dry_run:
            print(json.dumps(build_k8s_job_dry_run_output(manifest), ensure_ascii=False, indent=2))
            return

        try:
            from src.mlops.k8s_job_executor import create_k8s_job

            response = create_k8s_job(manifest)
            job_name = getattr(getattr(response, "metadata", None), "name", None) or _k8s_job_name(manifest)
            print(
                json.dumps(
                    {
                        "job_type": args.job_type,
                        "k8s_job_name": job_name,
                        "namespace": (manifest.get("metadata") or {}).get("namespace") or "admin",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        except Exception as exc:
            print(f"Failed to create Kubernetes Job: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.job_type != "training":
        raise ValueError(f"Unsupported job_type: {args.job_type}")

    missing = _missing_required(args)
    if args.dry_run:
        config = build_training_job_config(args, allow_placeholders=True)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "job_type": args.job_type,
                    "missing_required_for_real_run": missing,
                    "training_job_config": config,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if missing:
        raise ValueError(f"Missing required values: {', '.join(missing)}")

    config = build_training_job_config(args)
    result = _create_training_job(config, args.aws_region)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

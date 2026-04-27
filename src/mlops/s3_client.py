from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


REQUIRED_ARTIFACT_PATHS = (
    "sbert",
    "domain_model.pkl",
    "intent_model.pkl",
    "metrics.json",
    "config.json",
    "label_mapping.json",
)


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def _iter_files(local_dir: Path) -> Iterable[Path]:
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            yield path


def _s3_key(prefix: str, relative_path: Path) -> str:
    normalized_prefix = _normalize_prefix(prefix)
    relative_key = relative_path.as_posix()
    if not normalized_prefix:
        return relative_key
    return f"{normalized_prefix}/{relative_key}"


def plan_directory_upload(local_dir: str | Path, bucket: str, prefix: str) -> dict:
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {local_path}")
    if not local_path.is_dir():
        raise NotADirectoryError(f"Artifact path is not a directory: {local_path}")

    files = []
    for file_path in _iter_files(local_path):
        relative_path = file_path.relative_to(local_path)
        key = _s3_key(prefix, relative_path)
        files.append(
            {
                "local_path": str(file_path),
                "relative_path": relative_path.as_posix(),
                "bucket": bucket,
                "key": key,
                "s3_uri": f"s3://{bucket}/{key}",
            }
        )

    return {
        "local_dir": str(local_path),
        "bucket": bucket,
        "prefix": _normalize_prefix(prefix),
        "file_count": len(files),
        "files": files,
    }


def upload_directory_to_s3(local_dir: str | Path, bucket: str, prefix: str) -> dict:
    plan = plan_directory_upload(local_dir, bucket, prefix)

    import boto3

    s3_client = boto3.client("s3")
    uploaded_files = []
    for file_info in plan["files"]:
        s3_client.upload_file(file_info["local_path"], bucket, file_info["key"])
        uploaded_files.append(file_info)

    return {
        "bucket": bucket,
        "prefix": plan["prefix"],
        "uploaded_count": len(uploaded_files),
        "uploaded_files": uploaded_files,
    }


def upload_json_to_s3(payload: dict, bucket: str, key: str) -> dict:
    import boto3

    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )

    return {
        "bucket": bucket,
        "key": key,
        "s3_uri": f"s3://{bucket}/{key}",
        "content_type": "application/json",
        "bytes": len(body),
    }


def download_json_from_s3(bucket: str, key: str) -> dict:
    import boto3

    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def download_directory_from_s3(bucket: str, prefix: str, local_dir: str | Path) -> dict:
    import boto3

    normalized_prefix = _normalize_prefix(prefix)
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    list_prefix = f"{normalized_prefix}/" if normalized_prefix else ""

    downloaded_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue

            relative_key = key[len(normalized_prefix):].lstrip("/") if normalized_prefix else key
            destination = local_path / Path(relative_key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, key, str(destination))
            downloaded_files.append(
                {
                    "bucket": bucket,
                    "key": key,
                    "s3_uri": f"s3://{bucket}/{key}",
                    "local_path": str(destination),
                    "relative_path": Path(relative_key).as_posix(),
                }
            )

    return {
        "bucket": bucket,
        "prefix": normalized_prefix,
        "local_dir": str(local_path),
        "downloaded_count": len(downloaded_files),
        "downloaded_files": downloaded_files,
    }


def validate_model_artifact_dir(local_dir: str | Path) -> dict:
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Model artifact directory does not exist: {local_path}")
    if not local_path.is_dir():
        raise NotADirectoryError(f"Model artifact path is not a directory: {local_path}")

    missing_paths = []
    validated_paths = []
    for relative_name in REQUIRED_ARTIFACT_PATHS:
        path = local_path / relative_name
        if not path.exists():
            missing_paths.append(relative_name)
            continue
        if relative_name == "sbert" and not path.is_dir():
            missing_paths.append(f"{relative_name}/ (expected directory)")
            continue
        if relative_name != "sbert" and not path.is_file():
            missing_paths.append(f"{relative_name} (expected file)")
            continue
        validated_paths.append(relative_name)

    if missing_paths:
        missing = ", ".join(missing_paths)
        raise FileNotFoundError(
            f"Invalid model artifact directory: missing required artifact paths: {missing}"
        )

    files = [
        file_path.relative_to(local_path).as_posix()
        for file_path in _iter_files(local_path)
    ]

    return {
        "valid": True,
        "local_dir": str(local_path),
        "required_paths": list(REQUIRED_ARTIFACT_PATHS),
        "validated_paths": validated_paths,
        "file_count": len(files),
        "files": files,
    }

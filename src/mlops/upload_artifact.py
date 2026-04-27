from __future__ import annotations

import argparse
import json
from pathlib import Path

from .s3_client import (
    plan_directory_upload,
    upload_directory_to_s3,
    validate_model_artifact_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and upload a standard model artifact directory to S3."
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="Local model artifact directory produced by training_entrypoint.py.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Target S3 bucket name.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Target S3 prefix, for example models/v2026_04_27_001.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print upload mappings without uploading files.",
    )
    return parser.parse_args()


def build_upload_result(
    artifact_dir: Path,
    bucket: str,
    prefix: str,
    dry_run: bool,
) -> dict:
    validation = validate_model_artifact_dir(artifact_dir)

    if dry_run:
        upload_plan = plan_directory_upload(artifact_dir, bucket, prefix)
        return {
            "dry_run": True,
            "validation": validation,
            "upload_plan": upload_plan,
        }

    upload_result = upload_directory_to_s3(artifact_dir, bucket, prefix)
    return {
        "dry_run": False,
        "validation": validation,
        "upload_result": upload_result,
    }


def main() -> None:
    args = parse_args()
    result = build_upload_result(
        artifact_dir=args.artifact_dir,
        bucket=args.bucket,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

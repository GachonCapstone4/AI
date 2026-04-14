from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

from settings import get_settings


OPTIONAL_ARTIFACTS = ("label_mapping.json", "metadata.json")
REQUIRED_S3_FILES = (
    "domain_classifier.pkl",
    "intent_classifiers.pkl",
    "domain_label_encoder.pkl",
    "intent_label_encoders.pkl",
)
REQUIRED_SBERT_FILES = (
    "model.safetensors",
    "tokenizer.json",
)


@dataclass(frozen=True)
class RuntimeModelPaths:
    sbert_dir: Path
    sbert_model_path: Path
    sbert_tokenizer_path: Path
    domain_clf_path: Path
    domain_le_path: Path
    intent_clf_path: Path
    intent_le_path: Path
    label_mapping_path: Path | None = None
    metadata_path: Path | None = None


class S3ArtifactCache:
    def __init__(self, *, region_name: str) -> None:
        self._region_name = region_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "boto3 is required for MODEL_SOURCE=s3. Install project dependencies first."
                ) from exc

            self._client = boto3.client("s3", region_name=self._region_name)
        return self._client

    def download_file_if_missing(self, *, bucket: str, key: str, target_path: Path) -> None:
        if target_path.exists():
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._get_client().download_file(bucket, key, str(target_path))

    def download_prefix(self, *, bucket: str, prefix: str, target_dir: Path) -> None:
        paginator = self._get_client().get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        found = False
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                found = True
                relative_path = key[len(prefix):].lstrip("/")
                destination = target_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    continue
                self._get_client().download_file(bucket, key, str(destination))

        if not found:
            raise FileNotFoundError(f"No objects found under s3://{bucket}/{prefix}")

    def exists(self, *, bucket: str, key: str) -> bool:
        try:
            self._get_client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False


def _validate_required_local_paths(paths: RuntimeModelPaths) -> None:
    required = {
        "sbert/model.safetensors": paths.sbert_model_path,
        "sbert/tokenizer.json": paths.sbert_tokenizer_path,
        "domain_clf_path": paths.domain_clf_path,
        "domain_le_path": paths.domain_le_path,
        "intent_clf_path": paths.intent_clf_path,
        "intent_le_path": paths.intent_le_path,
    }

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        missing_items = ", ".join(missing)
        raise RuntimeError(f"Required model artifacts are missing: {missing_items}")


def _resolve_local_model_paths() -> RuntimeModelPaths:
    from config import (
        DOMAIN_CLF_PATH,
        DOMAIN_LE_PATH,
        INTENT_CLF_PATH,
        INTENT_LE_PATH,
        SBERT_MODEL_PATH,
    )

    paths = RuntimeModelPaths(
        sbert_dir=Path(SBERT_MODEL_PATH),
        sbert_model_path=Path(SBERT_MODEL_PATH) / "model.safetensors",
        sbert_tokenizer_path=Path(SBERT_MODEL_PATH) / "tokenizer.json",
        domain_clf_path=Path(DOMAIN_CLF_PATH),
        domain_le_path=Path(DOMAIN_LE_PATH),
        intent_clf_path=Path(INTENT_CLF_PATH),
        intent_le_path=Path(INTENT_LE_PATH),
    )
    _validate_required_local_paths(paths)
    return paths


def _build_s3_runtime_paths() -> RuntimeModelPaths:
    settings = get_settings()
    base = Path(settings.MODEL_LOCAL_CACHE_DIR) / settings.ACTIVE_MODEL_VERSION

    return RuntimeModelPaths(
        sbert_dir=base / "sbert",
        sbert_model_path=base / "sbert" / "model.safetensors",
        sbert_tokenizer_path=base / "sbert" / "tokenizer.json",
        domain_clf_path=base / "domain_classifier.pkl",
        domain_le_path=base / "domain_label_encoder.pkl",
        intent_clf_path=base / "intent_classifiers.pkl",
        intent_le_path=base / "intent_label_encoders.pkl",
        label_mapping_path=base / "label_mapping.json",
        metadata_path=base / "metadata.json",
    )


def _resolve_s3_model_paths() -> RuntimeModelPaths:
    settings = get_settings()
    cache = S3ArtifactCache(region_name=settings.AWS_REGION)

    paths = _build_s3_runtime_paths()
    cache_root = paths.domain_clf_path.parent
    base_prefix = f"{settings.S3_MODEL_PREFIX.rstrip('/')}/{settings.ACTIVE_MODEL_VERSION}"
    sbert_prefix = f"{base_prefix}/sbert/"

    if not all((paths.sbert_dir / filename).exists() for filename in REQUIRED_SBERT_FILES):
        cache.download_prefix(
            bucket=settings.S3_MODEL_BUCKET or "",
            prefix=sbert_prefix,
            target_dir=paths.sbert_dir,
        )

    required_files = {
        "domain_classifier.pkl": paths.domain_clf_path,
        "domain_label_encoder.pkl": paths.domain_le_path,
        "intent_classifiers.pkl": paths.intent_clf_path,
        "intent_label_encoders.pkl": paths.intent_le_path,
    }
    for filename, target_path in required_files.items():
        cache.download_file_if_missing(
            bucket=settings.S3_MODEL_BUCKET or "",
            key=f"{base_prefix}/{filename}",
            target_path=target_path,
        )

    for optional_name in OPTIONAL_ARTIFACTS:
        key = f"{base_prefix}/{optional_name}"
        target = cache_root / optional_name
        if cache.exists(bucket=settings.S3_MODEL_BUCKET or "", key=key):
            cache.download_file_if_missing(
                bucket=settings.S3_MODEL_BUCKET or "",
                key=key,
                target_path=target,
            )

    _validate_required_local_paths(paths)
    return paths


def resolve_runtime_model_paths() -> RuntimeModelPaths:
    settings = get_settings()
    if settings.MODEL_SOURCE == "s3":
        print(
            "[resolve_runtime_model_paths] source=s3 "
            f"base={Path(settings.MODEL_LOCAL_CACHE_DIR) / settings.ACTIVE_MODEL_VERSION}"
        )
        return _resolve_s3_model_paths()
    print("[resolve_runtime_model_paths] source=local")
    return _resolve_local_model_paths()


def load_classification_pipeline() -> dict:
    paths = resolve_runtime_model_paths()
    print(f"[load_classification_pipeline] sbert_dir={paths.sbert_dir}")

    pipeline = {
        "sbert": SentenceTransformer(str(paths.sbert_dir)),
        "domain_clf": joblib.load(str(paths.domain_clf_path)),
        "le_domain": joblib.load(str(paths.domain_le_path)),
        "intent_clf": joblib.load(str(paths.intent_clf_path)),
        "le_intent": joblib.load(str(paths.intent_le_path)),
    }

    if paths.metadata_path and paths.metadata_path.exists():
        pipeline["metadata"] = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    if paths.label_mapping_path and paths.label_mapping_path.exists():
        pipeline["label_mapping"] = json.loads(paths.label_mapping_path.read_text(encoding="utf-8"))

    print("[load_classification_pipeline] runtime model pipeline loaded")
    return pipeline

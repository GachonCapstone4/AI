from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

from settings import get_settings
from messaging.structured_log import get_logger


STANDARD_REQUIRED_FILES = (
    "sbert",
    "domain_model.pkl",
    "intent_model.pkl",
    "label_mapping.json",
    "metrics.json",
    "config.json",
)
COMPLETE_MARKER = ".complete"
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

log = get_logger("model_loader")


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
    """Resolve legacy local artifact names for dev/local mode only."""
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
    raise RuntimeError(
        "Legacy S3 runtime path resolver is disabled. "
        "MODEL_SOURCE=s3 must use latest.json/ACTIVE_MODEL_VERSION with standard SageMaker artifacts."
    )


def _resolve_s3_model_paths() -> RuntimeModelPaths:
    return _build_s3_runtime_paths()


def resolve_runtime_model_paths() -> RuntimeModelPaths:
    settings = get_settings()
    if settings.MODEL_SOURCE == "s3":
        raise RuntimeError(
            "resolve_runtime_model_paths is local/dev-only. "
            "MODEL_SOURCE=s3 startup must call load_standard_model_bundle."
        )
    log.info("runtime_model_paths_resolved", model_source="local", artifact_format="legacy_dev")
    return _resolve_local_model_paths()


def _model_cache_dir_for_version(version: str) -> Path:
    settings = get_settings()
    return Path(settings.MODEL_LOCAL_CACHE_DIR) / version


def parse_latest_model_version(payload: dict) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError("models/latest.json must be a JSON object.")

    snake_version = payload.get("model_version")
    camel_version = payload.get("modelVersion")
    if snake_version and camel_version and snake_version != camel_version:
        raise RuntimeError(
            "models/latest.json contains conflicting model_version and modelVersion values."
        )

    model_version = snake_version or camel_version
    if not isinstance(model_version, str) or not model_version.strip():
        raise RuntimeError(
            "models/latest.json must contain a non-empty model_version "
            "(modelVersion is accepted only as a compatibility alias)."
        )
    return model_version.strip().strip("/")


def load_latest_model_version() -> str:
    settings = get_settings()
    if not settings.S3_MODEL_BUCKET:
        raise RuntimeError("S3_MODEL_BUCKET is required to load models/latest.json")

    cache = S3ArtifactCache(region_name=settings.AWS_REGION)
    latest_key = f"{settings.S3_MODEL_PREFIX.rstrip('/')}/latest.json"
    response = cache._get_client().get_object(
        Bucket=settings.S3_MODEL_BUCKET,
        Key=latest_key,
    )
    payload = json.loads(response["Body"].read().decode("utf-8"))
    return parse_latest_model_version(payload)


def ensure_standard_model_artifact_cached(version: str) -> Path:
    settings = get_settings()
    artifact_dir = _model_cache_dir_for_version(version)
    complete_marker = artifact_dir / COMPLETE_MARKER

    if settings.MODEL_SOURCE == "s3":
        if complete_marker.exists():
            _validate_standard_artifact_dir(artifact_dir)
            return artifact_dir

        cache_root = artifact_dir.parent
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = cache_root / f"{version}.tmp-{uuid.uuid4().hex}"
        cache = S3ArtifactCache(region_name=settings.AWS_REGION)
        base_prefix = f"{settings.S3_MODEL_PREFIX.rstrip('/')}/{version}"
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            cache.download_prefix(
                bucket=settings.S3_MODEL_BUCKET or "",
                prefix=f"{base_prefix}/",
                target_dir=tmp_dir,
            )
            _validate_standard_artifact_dir(tmp_dir)
            (tmp_dir / COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")

            if complete_marker.exists():
                _validate_standard_artifact_dir(artifact_dir)
                return artifact_dir
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            os.replace(tmp_dir, artifact_dir)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    _validate_standard_artifact_dir(artifact_dir)
    return artifact_dir


def _validate_standard_artifact_dir(artifact_dir: Path) -> None:
    missing = []
    for name in STANDARD_REQUIRED_FILES:
        path = artifact_dir / name
        if not path.exists():
            missing.append(path)
        elif name == "sbert" and not path.is_dir():
            missing.append(path)
        elif name != "sbert" and not path.is_file():
            missing.append(path)
    for filename in REQUIRED_SBERT_FILES:
        path = artifact_dir / "sbert" / filename
        if not path.exists() or not path.is_file():
            missing.append(path)
    if missing:
        missing_items = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Standard model artifacts are missing: {missing_items}")


def load_standard_model_bundle(version: str) -> dict:
    settings = get_settings()
    artifact_dir = ensure_standard_model_artifact_cached(version)

    domain_payload = joblib.load(str(artifact_dir / "domain_model.pkl"))
    intent_payload = joblib.load(str(artifact_dir / "intent_model.pkl"))
    label_mapping = json.loads(
        (artifact_dir / "label_mapping.json").read_text(encoding="utf-8")
    )
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))

    bundle = {
        "sbert": SentenceTransformer(str(artifact_dir / "sbert")),
        "domain_clf": domain_payload["classifier"],
        "le_domain": domain_payload["label_encoder"],
        "intent_clf": intent_payload["classifiers"],
        "le_intent": intent_payload["label_encoders"],
        "label_mapping": label_mapping,
        "config": config,
        "metrics": metrics,
        "runtime": {
            "model_source": settings.MODEL_SOURCE,
            "active_model_version": version,
            "loaded_sbert_path": str(artifact_dir / "sbert"),
            "loaded_domain_model_path": str(artifact_dir / "domain_model.pkl"),
            "loaded_intent_model_path": str(artifact_dir / "intent_model.pkl"),
            "loaded_config_path": str(artifact_dir / "config.json"),
            "loaded_metrics_path": str(artifact_dir / "metrics.json"),
            "loaded_label_mapping_path": str(artifact_dir / "label_mapping.json"),
            "metadata_model_version": config.get("model_version"),
            "artifact_format": "standard",
        },
    }

    log.info(
        "standard_model_bundle_loaded",
        model_source=settings.MODEL_SOURCE,
        active_model_version=version,
        loaded_sbert_path=bundle["runtime"]["loaded_sbert_path"],
        loaded_domain_model_path=bundle["runtime"]["loaded_domain_model_path"],
        loaded_intent_model_path=bundle["runtime"]["loaded_intent_model_path"],
    )
    return bundle


def load_classification_pipeline() -> dict:
    settings = get_settings()
    if settings.MODEL_SOURCE == "s3":
        raise RuntimeError(
            "load_classification_pipeline is local/dev-only and uses legacy artifact names. "
            "Use load_standard_model_bundle for MODEL_SOURCE=s3."
        )
    paths = resolve_runtime_model_paths()

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

    metadata = pipeline.get("metadata") or {}
    runtime = {
        "model_source": settings.MODEL_SOURCE,
        "active_model_version": settings.ACTIVE_MODEL_VERSION,
        "loaded_sbert_path": str(paths.sbert_dir),
        "loaded_domain_model_path": str(paths.domain_clf_path),
        "loaded_domain_label_encoder_path": str(paths.domain_le_path),
        "loaded_intent_model_path": str(paths.intent_clf_path),
        "loaded_intent_label_encoder_path": str(paths.intent_le_path),
        "loaded_metadata_path": str(paths.metadata_path) if paths.metadata_path else None,
        "loaded_label_mapping_path": str(paths.label_mapping_path) if paths.label_mapping_path else None,
        "metadata_model_version": metadata.get("model_version") or metadata.get("modelVersion"),
    }
    pipeline["runtime"] = runtime

    log.info(
        "classification_pipeline_loaded",
        model_source=runtime["model_source"],
        active_model_version=runtime["active_model_version"],
        metadata_model_version=runtime["metadata_model_version"],
        loaded_sbert_path=runtime["loaded_sbert_path"],
        loaded_domain_model_path=runtime["loaded_domain_model_path"],
        loaded_intent_model_path=runtime["loaded_intent_model_path"],
    )
    return pipeline

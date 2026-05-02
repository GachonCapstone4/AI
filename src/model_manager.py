from __future__ import annotations

import sys
from threading import RLock
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .inference import predict_email
    from .model_loader import (
        load_classification_pipeline,
        load_latest_model_version,
        load_standard_model_bundle,
    )
    from .settings import get_settings
except ImportError:
    from inference import predict_email
    from model_loader import (
        load_classification_pipeline,
        load_latest_model_version,
        load_standard_model_bundle,
    )
    from settings import get_settings


VALIDATION_SAMPLES = (
    "Please review the attached invoice and confirm payment status.",
    "Can you help me reset access to my account?",
)


class ModelManager:
    def __init__(self) -> None:
        self.current_bundle: dict | None = None
        self.staging_bundle: dict | None = None
        self.current_model_version: str | None = None
        self.staging_model_version: str | None = None
        self.staging_validated = False
        self._lock = RLock()

    def _resolve_requested_version(self, version: str | None = None) -> str:
        if version:
            return version
        settings = get_settings()
        if settings.ACTIVE_MODEL_VERSION:
            return settings.ACTIVE_MODEL_VERSION
        return load_latest_model_version()

    def load_initial_model(self, existing_bundle: dict | None = None) -> dict:
        settings = get_settings()
        if existing_bundle is None:
            if settings.MODEL_SOURCE == "s3":
                model_version = self._resolve_requested_version()
                existing_bundle = load_standard_model_bundle(model_version)
            else:
                existing_bundle = load_classification_pipeline()

        runtime = existing_bundle.get("runtime") or {}
        model_version = (
            runtime.get("active_model_version")
            or runtime.get("metadata_model_version")
            or settings.ACTIVE_MODEL_VERSION
        )
        with self._lock:
            self.current_bundle = existing_bundle
            self.current_model_version = model_version
            self.staging_bundle = None
            self.staging_model_version = None
            self.staging_validated = False
        return {
            "status": "ok",
            "model_version": model_version,
            "runtime": runtime,
        }

    def predict(self, text: str) -> dict:
        with self._lock:
            bundle = self.current_bundle
        if bundle is None:
            raise RuntimeError("Current model bundle is not loaded.")
        return predict_email(text, bundle)

    def preload(self, version: str | None = None) -> dict:
        with self._lock:
            self.staging_validated = False
        version = self._resolve_requested_version(version)
        bundle = load_standard_model_bundle(version)
        with self._lock:
            self.staging_bundle = bundle
            self.staging_model_version = version
            self.staging_validated = False
        return {
            "status": "preloaded",
            "model_version": version,
            "artifact_s3_uri": self._artifact_s3_uri(version),
            "runtime": bundle.get("runtime") or {},
        }

    def validate(self) -> dict:
        with self._lock:
            bundle = self.staging_bundle
            model_version = self.staging_model_version
            self.staging_validated = False
        if bundle is None or model_version is None:
            raise RuntimeError("No staging model bundle is loaded.")

        domains, intents_by_domain = self._validate_label_mapping(bundle.get("label_mapping"))

        predictions: list[dict[str, Any]] = []
        for sample in VALIDATION_SAMPLES:
            prediction = predict_email(sample, bundle)
            domain = prediction.get("domain")
            intent = prediction.get("intent")

            if not domain or not intent:
                raise RuntimeError("Validation prediction did not include domain and intent.")
            if domain not in domains:
                raise RuntimeError(f"Predicted domain is not in label_mapping: {domain}")
            valid_intents = intents_by_domain.get(domain) or []
            if not valid_intents or intent not in valid_intents:
                raise RuntimeError(
                    f"Predicted intent is not in label_mapping for domain {domain}: {intent}"
                )

            predictions.append(
                {
                    "sample": sample,
                    "domain": domain,
                    "intent": intent,
                    "confidence_score": prediction.get("confidence_score"),
                }
            )

        with self._lock:
            if self.staging_bundle is bundle and self.staging_model_version == model_version:
                self.staging_validated = True

        return {
            "status": "validated",
            "model_version": model_version,
            "samples": predictions,
        }

    def switch(self) -> dict:
        with self._lock:
            if self.staging_bundle is None or self.staging_model_version is None:
                raise RuntimeError("No staging model bundle is loaded.")
            if not self.staging_validated:
                raise RuntimeError("Staging model bundle has not been validated.")

            self.current_bundle = self.staging_bundle
            self.current_model_version = self.staging_model_version
            model_version = self.current_model_version
            self.staging_bundle = None
            self.staging_model_version = None
            self.staging_validated = False

        return {
            "status": "switched",
            "model_version": model_version,
        }

    def _artifact_s3_uri(self, version: str) -> str | None:
        settings = get_settings()
        if not settings.S3_MODEL_BUCKET:
            return None
        prefix = settings.S3_MODEL_PREFIX.rstrip("/")
        return f"s3://{settings.S3_MODEL_BUCKET}/{prefix}/{version}/"

    def _validate_label_mapping(self, label_mapping: Any) -> tuple[set[str], dict[str, list[str]]]:
        if not isinstance(label_mapping, dict):
            raise RuntimeError("label_mapping.json must be a JSON object.")

        raw_domains = label_mapping.get("domains")
        raw_intents = label_mapping.get("intents")
        if not isinstance(raw_domains, list) or not raw_domains:
            raise RuntimeError("label_mapping.json must include a non-empty domains list.")
        if not isinstance(raw_intents, dict) or not raw_intents:
            raise RuntimeError("label_mapping.json must include a non-empty intents object.")

        domains = {item for item in raw_domains if isinstance(item, str) and item}
        if len(domains) != len(raw_domains):
            raise RuntimeError("label_mapping.json domains must contain only non-empty strings.")

        intents_by_domain: dict[str, list[str]] = {}
        for domain in domains:
            intents = raw_intents.get(domain)
            if not isinstance(intents, list) or not intents:
                raise RuntimeError(
                    f"label_mapping.json intents must include a non-empty list for domain {domain}."
                )
            valid_intents = [item for item in intents if isinstance(item, str) and item]
            if len(valid_intents) != len(intents):
                raise RuntimeError(
                    f"label_mapping.json intents for domain {domain} must contain only non-empty strings."
                )
            intents_by_domain[domain] = valid_intents

        return domains, intents_by_domain

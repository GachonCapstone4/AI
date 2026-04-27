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
    from .model_loader import load_classification_pipeline, load_standard_model_bundle
    from .settings import get_settings
except ImportError:
    from inference import predict_email
    from model_loader import load_classification_pipeline, load_standard_model_bundle
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
        self._lock = RLock()

    def load_initial_model(self, existing_bundle: dict | None = None) -> dict:
        if existing_bundle is None:
            existing_bundle = load_classification_pipeline()

        runtime = existing_bundle.get("runtime") or {}
        settings = get_settings()
        model_version = (
            runtime.get("active_model_version")
            or runtime.get("metadata_model_version")
            or settings.ACTIVE_MODEL_VERSION
        )
        with self._lock:
            self.current_bundle = existing_bundle
            self.current_model_version = model_version
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

    def preload(self, version: str) -> dict:
        bundle = load_standard_model_bundle(version)
        with self._lock:
            self.staging_bundle = bundle
            self.staging_model_version = version
        return {
            "status": "ok",
            "model_version": version,
            "runtime": bundle.get("runtime") or {},
        }

    def validate(self) -> dict:
        with self._lock:
            bundle = self.staging_bundle
            model_version = self.staging_model_version
        if bundle is None or model_version is None:
            raise RuntimeError("No staging model bundle is loaded.")

        label_mapping = bundle.get("label_mapping") or {}
        domains = set(label_mapping.get("domains") or [])
        intents_by_domain = label_mapping.get("intents") or {}

        predictions: list[dict[str, Any]] = []
        for sample in VALIDATION_SAMPLES:
            prediction = predict_email(sample, bundle)
            domain = prediction.get("domain")
            intent = prediction.get("intent")

            if not domain or not intent:
                raise RuntimeError("Validation prediction did not include domain and intent.")
            if domains and domain not in domains:
                raise RuntimeError(f"Predicted domain is not in label_mapping: {domain}")
            if intents_by_domain:
                valid_intents = intents_by_domain.get(domain) or []
                if valid_intents and intent not in valid_intents:
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

        return {
            "status": "ok",
            "model_version": model_version,
            "predictions": predictions,
        }

    def switch(self) -> dict:
        with self._lock:
            if self.staging_bundle is None or self.staging_model_version is None:
                raise RuntimeError("No staging model bundle is loaded.")

            self.current_bundle = self.staging_bundle
            self.current_model_version = self.staging_model_version
            model_version = self.current_model_version
            self.staging_bundle = None
            self.staging_model_version = None

        return {
            "status": "ok",
            "model_version": model_version,
        }

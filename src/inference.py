import numpy as np

from config import CONFIDENCE_THRESHOLD
from settings import get_settings
from model_loader import (
    load_classification_pipeline,
    load_latest_model_version,
    load_standard_model_bundle,
)


def load_classify_pipeline() -> dict:
    settings = get_settings()
    if settings.MODEL_SOURCE == "s3":
        version = settings.ACTIVE_MODEL_VERSION or load_latest_model_version()
        return load_standard_model_bundle(version)
    return load_classification_pipeline()


def load_pipeline() -> dict:
    return load_classify_pipeline()


def predict_email(
    email_text: str,
    pipeline: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    user_domain: str = None,
) -> dict:
    emb = pipeline["sbert"].encode([email_text], normalize_embeddings=True)

    domain_classes = pipeline["le_domain"].classes_

    if user_domain and user_domain in pipeline["intent_clf"]:
        domain_name = user_domain
        domain_conf = 1.0
        domain_source = "onboarding"
        top2_domains = [{"domain": domain_name, "confidence": 1.0}]
    else:
        domain_proba = pipeline["domain_clf"].predict_proba(emb)[0]
        top2_indices = np.argsort(domain_proba)[::-1][:2]

        top2_domains = [
            {
                "domain": domain_classes[idx],
                "confidence": round(float(domain_proba[idx]), 4),
            }
            for idx in top2_indices
        ]

        domain_idx = top2_indices[0]
        domain_conf = domain_proba[domain_idx]
        domain_name = domain_classes[domain_idx]
        domain_source = "classifier"

    intent_name, intent_conf = "unknown", 0.0

    if domain_name in pipeline["intent_clf"]:
        intent_proba = pipeline["intent_clf"][domain_name].predict_proba(emb)[0]
        intent_idx = np.argmax(intent_proba)
        intent_conf = intent_proba[intent_idx]
        intent_name = pipeline["le_intent"][domain_name].inverse_transform([intent_idx])[0]

    low_confidence = (
        float(domain_conf) < confidence_threshold
        or float(intent_conf) < confidence_threshold
    )
    confidence_score = min(float(domain_conf), float(intent_conf))

    return {
        "domain": domain_name,
        "domain_confidence": round(float(domain_conf), 4),
        "intent": intent_name,
        "intent_confidence": round(float(intent_conf), 4),
        "confidence_score": round(confidence_score, 4),
        "low_confidence": low_confidence,
        "top2_domains": top2_domains,
        "domain_source": domain_source,
    }


def predict_batch(
    email_texts: list,
    pipeline: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    user_domain: str = None,
) -> list:
    return [
        predict_email(t, pipeline, confidence_threshold, user_domain)
        for t in email_texts
    ]

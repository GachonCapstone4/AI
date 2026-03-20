# ============================================================
# 추론 파이프라인 (Fallback + Top-2 도메인 반환 포함)
# FastAPI 등 백엔드 연동 시 이 파일만 import
# ============================================================

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

from config import (
    SBERT_MODEL_PATH,
    DOMAIN_CLF_PATH, DOMAIN_LE_PATH,
    INTENT_CLF_PATH, INTENT_LE_PATH,
    CONFIDENCE_THRESHOLD,
)


# ── 파이프라인 전체 로드 ────────────────────────────────────
def load_pipeline(
    sbert_path=SBERT_MODEL_PATH,
    domain_clf_path=DOMAIN_CLF_PATH,
    domain_le_path=DOMAIN_LE_PATH,
    intent_clf_path=INTENT_CLF_PATH,
    intent_le_path=INTENT_LE_PATH,
) -> dict:
    """저장된 모델 전체 로드 → pipeline dict"""

    pipeline = {
        "sbert": SentenceTransformer(str(sbert_path)),
        "domain_clf": joblib.load(str(domain_clf_path)),
        "le_domain": joblib.load(str(domain_le_path)),
        "intent_clf": joblib.load(str(intent_clf_path)),
        "le_intent": joblib.load(str(intent_le_path)),
    }

    print("[load_pipeline] 파이프라인 로드 완료")
    return pipeline


# ── 단일 이메일 추론 ────────────────────────────────────────
def predict_email(
    email_text: str,
    pipeline: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    user_domain: str = None,  # 온보딩에서 받은 도메인
) -> dict:
    """
    email_text → domain + intent 예측

    user_domain 있을 때 (온보딩 도메인 고정):
        → Domain 분류기 스킵
        → 해당 도메인 Intent 분류기만 사용
        → domain_confidence = 1.0

    user_domain 없을 때:
        → Domain 분류기 → Intent 분류기 순서대로 실행
        → confidence < threshold 시 low_confidence = True
        → Top-2 도메인 반환

    return:
        {
            domain, domain_confidence,
            intent, intent_confidence,
            low_confidence,
            top2_domains,
            domain_source  ← "onboarding" or "classifier"
        }
    """
    emb = pipeline["sbert"].encode([email_text], normalize_embeddings=True)

    # ── 1차: Domain 결정 ─────────────────────────────────────
    domain_classes = pipeline["le_domain"].classes_

    if user_domain and user_domain in pipeline["intent_clf"]:
        # 온보딩 도메인 사용 → 분류기 스킵
        domain_name = user_domain
        domain_conf = 1.0
        domain_source = "onboarding"
        top2_domains = [{"domain": domain_name, "confidence": 1.0}]

    else:
        # 온보딩 도메인 없음 → 분류기 사용
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

    # ── 2차: Intent 예측 (domain 조건부) ─────────────────────
    intent_name, intent_conf = "unknown", 0.0

    if domain_name in pipeline["intent_clf"]:
        intent_proba = pipeline["intent_clf"][domain_name].predict_proba(emb)[0]
        intent_idx = np.argmax(intent_proba)
        intent_conf = intent_proba[intent_idx]
        intent_name = pipeline["le_intent"][domain_name].inverse_transform([intent_idx])[0]

    # ── Fallback 판정 ─────────────────────────────────────────
    # 온보딩 도메인 사용 시 domain_conf = 1.0 이므로 intent만 체크
    low_confidence = (
        float(domain_conf) < confidence_threshold
        or float(intent_conf) < confidence_threshold
    )

    return {
        "domain": domain_name,
        "domain_confidence": round(float(domain_conf), 4),
        "intent": intent_name,
        "intent_confidence": round(float(intent_conf), 4),
        "low_confidence": low_confidence,
        "top2_domains": top2_domains,
        "domain_source": domain_source,  # 온보딩인지 분류기인지 확인용
    }


# ── 배치 추론 ───────────────────────────────────────────────
def predict_batch(
    email_texts: list,
    pipeline: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    user_domain: str = None,
) -> list:
    """List[str] → List[dict]"""
    return [
        predict_email(t, pipeline, confidence_threshold, user_domain)
        for t in email_texts
    ]
# ============================================================
# Intent Classifier 학습 + 평가 + 저장 (Domain-conditional)
# 모델 → models/  /  Confusion Matrix → outputs/figures/
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from config import (
    MODEL_DIR,
    LR_MAX_ITER, LR_C, LR_SOLVER,
    INTENT_CLF_PATH, INTENT_LE_PATH,
)
from evaluation import evaluate_classifier


def train_intent_classifiers(
    X  : np.ndarray,
    df : pd.DataFrame,
    model_path = INTENT_CLF_PATH,
    label_encoders_path = INTENT_LE_PATH,
) -> tuple:
    """
    도메인별 Intent LR 학습 + 평가 + models/ 저장
    return: (intent_classifiers, intent_encoders)
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    intent_classifiers = {}
    intent_encoders    = {}

    print("=" * 50)
    print("Intent Classifier — 도메인별 학습 및 평가")
    print("=" * 50)

    for domain in df["domain"].unique():
        mask            = df["domain"] == domain
        X_domain        = X[mask]
        y_intent_domain = df.loc[mask, "intent"].values
        unique_intents  = np.unique(y_intent_domain)

        if len(unique_intents) < 2:
            print(f"[SKIP] {domain}: 인텐트 수 < 2")
            continue

        le_intent = LabelEncoder()
        y_enc     = le_intent.fit_transform(y_intent_domain)

        clf = LogisticRegression(
            max_iter    =LR_MAX_ITER,
            C           =LR_C,
            solver      =LR_SOLVER,
            multi_class ="multinomial",
            random_state=42,
        )

        n_splits  = min(5, np.bincount(y_enc).min())
        safe_name = domain.replace("/", "_").replace(" ", "_")

        print(f"\n[{domain}] 샘플: {len(y_enc)} | 인텐트: {len(unique_intents)} | K={n_splits}")

        if n_splits >= 2:
            # 평가 (Confusion Matrix → outputs/figures/)
            evaluate_classifier(
                clf         =clf,
                X           =X_domain,
                y_enc       =y_enc,
                label_names =le_intent.classes_.tolist(),
                title       =f"Intent [{domain}]",
                fig_filename=f"intent_cm_{safe_name}.png",
                cmap        ="Greens",
                n_splits    =n_splits,
            )

        # 전체 데이터 최종 학습
        clf.fit(X_domain, y_enc)
        intent_classifiers[domain] = clf
        intent_encoders[domain]    = le_intent

    # models/ 저장
    joblib.dump(intent_classifiers, model_path)
    if label_encoders_path is not None:
        joblib.dump(intent_encoders, label_encoders_path)
    print(f"\n[train_intent] 저장 완료 → {os.path.dirname(model_path)}")

    return intent_classifiers, intent_encoders

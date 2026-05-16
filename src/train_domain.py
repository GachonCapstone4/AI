# ============================================================
# Domain Classifier 학습 + 평가 + 저장
# 모델 → models/  /  Confusion Matrix → outputs/figures/
# ============================================================

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from config import (
    MODEL_DIR,
    LR_MAX_ITER, LR_C, LR_SOLVER, LR_KFOLD,
    DOMAIN_CLF_PATH, DOMAIN_LE_PATH,
)
from evaluation import evaluate_classifier


def train_domain_classifier(
    X        : np.ndarray,
    y_domain : np.ndarray,
    model_path = DOMAIN_CLF_PATH,
    label_encoder_path = DOMAIN_LE_PATH,
    return_metadata: bool = False,
) -> tuple:
    """
    Domain LR 학습 + 평가 + models/ 저장
    return: (domain_clf, le_domain)
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    le_domain    = LabelEncoder()
    y_domain_enc = le_domain.fit_transform(y_domain)
    label_distribution = {
        str(label): int(count)
        for label, count in zip(*np.unique(y_domain, return_counts=True))
    }
    print(f"[train_domain] 총 샘플: {len(y_domain)}", flush=True)
    print(f"[train_domain] domain label distribution: {label_distribution}", flush=True)

    if len(le_domain.classes_) < 2:
        raise ValueError(
            "Domain classifier training requires at least 2 domain classes; "
            f"found {len(le_domain.classes_)} class with distribution={label_distribution}"
        )

    clf = LogisticRegression(
        max_iter    =LR_MAX_ITER,
        C           =LR_C,
        solver      =LR_SOLVER,
        multi_class ="multinomial",
        random_state=42,
    )

    # 평가 (Confusion Matrix → outputs/figures/)
    cv_evaluation = evaluate_classifier(
        clf         =clf,
        X           =X,
        y_enc       =y_domain_enc,
        label_names =le_domain.classes_.tolist(),
        title       ="Domain Classifier",
        fig_filename="domain_confusion_matrix.png",
        cmap        ="Blues",
        n_splits    =LR_KFOLD,
    )

    # 전체 데이터 최종 학습
    clf.fit(X, y_domain_enc)

    # models/ 저장
    joblib.dump(clf, model_path)
    if label_encoder_path is not None:
        joblib.dump(le_domain, label_encoder_path)
    print(f"[train_domain] 저장 완료 → {os.path.dirname(model_path)}")

    if return_metadata:
        return clf, le_domain, {
            "domain_label_distribution": label_distribution,
            "domain_cv": cv_evaluation,
        }
    return clf, le_domain

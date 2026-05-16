# ============================================================
# 임베딩 품질 검증 / F1-score / Confusion Matrix / ROC Curve
# Confusion Matrix, ROC → outputs/figures/ 저장
# 실행: cd src && python evaluation.py
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics.pairwise import cosine_similarity

# src/ 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    FIGURES_DIR,
    LR_KFOLD,
    DOMAIN_CLF_PATH, DOMAIN_LE_PATH,
    INTENT_CLF_PATH, INTENT_LE_PATH,
    EMBEDDINGS_FINETUNED_PATH,
    DATASET_PATH,
)


# ── 한글 폰트 설정 (Windows / Linux / Mac 자동 감지) ────────
def _set_korean_font() -> None:
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
        sns.set_theme(font="Malgun Gothic")
        print("[폰트] Windows - 맑은 고딕 설정 완료")
    elif system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
        plt.rcParams["axes.unicode_minus"] = False
        sns.set_theme(font="AppleGothic")
        print("[폰트] Mac - AppleGothic 설정 완료")
    else:
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            sns.set_theme(font=font_name)
            print(f"[폰트] Linux - {font_name} 설정 완료")
        else:
            print("[경고] NanumGothic 폰트 없음. 한글이 깨질 수 있습니다.")

_set_korean_font()


def _label_distribution(y_enc: np.ndarray, label_names: list) -> dict:
    values, counts = np.unique(y_enc, return_counts=True)
    return {
        str(label_names[int(value)] if int(value) < len(label_names) else value): int(count)
        for value, count in zip(values, counts)
    }


def _cv_skip_result(title: str, reason: str, label_distribution: dict, n_splits: int) -> dict:
    print(f"\n[{title}] K-Fold 평가 스킵: {reason}", flush=True)
    print(f"[{title}] label distribution: {label_distribution}", flush=True)
    return {
        "cv_skipped": True,
        "cv_skip_reason": reason,
        "label_distribution": label_distribution,
        "requested_n_splits": int(n_splits),
        "effective_n_splits": None,
        "weighted_f1_mean": None,
        "weighted_f1_std": None,
        "macro_f1_mean": None,
        "macro_f1_std": None,
    }


# ── 1. 임베딩 품질 검증 ─────────────────────────────────────
def validate_embeddings(X: np.ndarray, df: pd.DataFrame) -> None:
    """SBERT 임베딩 품질 검증 — cosine similarity / mean / std"""
    print("\n" + "=" * 50)
    print("SBERT 임베딩 품질 검증")
    print("=" * 50)
    print(f"  Shape : {X.shape} | Mean: {np.mean(X):.4f} | Std: {np.std(X):.4f}")

    same_mask = df["intent"] == df["intent"].iloc[0]
    same_idx  = df[same_mask].index[:2].tolist()
    diff_idx  = df[~same_mask].index[0]

    print(f"\n[Cosine Similarity 샘플 검증]")
    print(f"  같은 Intent : {cosine_similarity(X[same_idx[0]].reshape(1,-1), X[same_idx[1]].reshape(1,-1))[0][0]:.4f}  ← 높을수록 좋음")
    print(f"  다른 Intent : {cosine_similarity(X[same_idx[0]].reshape(1,-1), X[diff_idx].reshape(1,-1))[0][0]:.4f}  ← 낮을수록 좋음")
    print(f"  랜덤 (0,100): {cosine_similarity(X[0].reshape(1,-1), X[100].reshape(1,-1))[0][0]:.4f}")

    print(f"\n[도메인별 Intra-class 평균 Cosine Similarity]")
    for domain in df["domain"].unique():
        idxs = df[df["domain"] == domain].index.tolist()[:20]
        if len(idxs) < 2:
            continue
        sims = cosine_similarity(X[idxs])
        np.fill_diagonal(sims, np.nan)
        print(f"  {domain:<25}: {np.nanmean(sims):.4f}")


# ── 2. 분류기 평가 (F1 + Confusion Matrix) ──────────────────
def evaluate_classifier(
    clf,
    X           : np.ndarray,
    y_enc       : np.ndarray,
    label_names : list,
    title       : str,
    fig_filename: str,
    cmap        : str = "Blues",
    n_splits    : int = LR_KFOLD,
) -> dict:
    """K-Fold F1 + Confusion Matrix → outputs/figures/ 저장"""
    y_enc = np.asarray(y_enc)
    label_distribution = _label_distribution(y_enc, label_names)
    class_count = len(label_distribution)

    print(f"\n[{title}] label distribution: {label_distribution}", flush=True)
    if class_count < 2:
        return _cv_skip_result(
            title,
            "class_count < 2",
            label_distribution,
            n_splits,
        )

    min_class_count = min(label_distribution.values())
    if min_class_count < 2:
        return _cv_skip_result(
            title,
            "min_class_count < 2",
            label_distribution,
            n_splits,
        )

    effective_n_splits = min(int(n_splits), int(min_class_count))
    if effective_n_splits != n_splits:
        print(
            f"[{title}] n_splits 조정: requested={n_splits}, effective={effective_n_splits} "
            f"(min_class_count={min_class_count})",
            flush=True,
        )

    cv = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=42)

    try:
        cv_w = cross_val_score(clf, X, y_enc, cv=cv, scoring="f1_weighted")
        cv_m = cross_val_score(clf, X, y_enc, cv=cv, scoring="f1_macro")
        y_pred = cross_val_predict(clf, X, y_enc, cv=cv)
    except Exception as exc:
        return _cv_skip_result(
            title,
            f"cross_validation_failed: {exc}",
            label_distribution,
            n_splits,
        )

    print(f"\n[{title}] K-Fold 결과")
    print(f"  Weighted F1 : {cv_w.mean():.4f} ± {cv_w.std():.4f}")
    print(f"  Macro F1    : {cv_m.mean():.4f} ± {cv_m.std():.4f}")

    print(f"\n[{title}] Classification Report")
    print(classification_report(y_enc, y_pred, target_names=label_names, zero_division=0))

    cm        = confusion_matrix(y_enc, y_pred)
    fig_h     = max(5, len(label_names))
    save_path = os.path.join(FIGURES_DIR, fig_filename)

    plt.figure(figsize=(fig_h + 2, fig_h))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5
    )
    plt.title(f"{title} — Confusion Matrix", fontsize=13)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion Matrix 저장 → {save_path}")
    return {
        "cv_skipped": False,
        "cv_skip_reason": None,
        "label_distribution": label_distribution,
        "requested_n_splits": int(n_splits),
        "effective_n_splits": int(effective_n_splits),
        "weighted_f1_mean": round(float(cv_w.mean()), 4),
        "weighted_f1_std": round(float(cv_w.std()), 4),
        "macro_f1_mean": round(float(cv_m.mean()), 4),
        "macro_f1_std": round(float(cv_m.std()), 4),
    }


# ── 3. ROC Curve (One-vs-Rest) ──────────────────────────────
def plot_roc_curve(
    clf,
    X           : np.ndarray,
    y_enc       : np.ndarray,
    label_names : list,
    title       : str,
    fig_filename: str,
    n_splits    : int = LR_KFOLD,
) -> None:
    """One-vs-Rest ROC Curve 시각화 + AUC 출력 → outputs/figures/ 저장"""
    n_classes = len(label_names)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_prob = cross_val_predict(clf, X, y_enc, cv=cv, method="predict_proba")
    y_bin  = label_binarize(y_enc, classes=list(range(n_classes)))

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    all_fpr  = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    colors = plt.cm.get_cmap("tab10", n_classes)
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, name in enumerate(label_names):
        ax.plot(
            fpr_dict[i], tpr_dict[i],
            color=colors(i), lw=1.5, alpha=0.75,
            label=f"{name} (AUC = {auc_dict[i]:.3f})"
        )

    ax.plot(all_fpr, mean_tpr, color="black", lw=2.5, linestyle="--",
            label=f"Macro-avg (AUC = {macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":", label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title(f"{title} — ROC Curve (One-vs-Rest)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, fig_filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n[{title}] ROC AUC 결과")
    for i, name in enumerate(label_names):
        print(f"  {name:<25}: AUC = {auc_dict[i]:.4f}")
    print(f"  {'Macro Average':<25}: AUC = {macro_auc:.4f}")
    print(f"ROC Curve 저장 → {save_path}")


# ── 실행 진입점 ──────────────────────────────────────────────
if __name__ == "__main__":

    # figures 폴더 없으면 생성
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    X  = np.load(str(EMBEDDINGS_FINETUNED_PATH))
    df = pd.read_csv(str(DATASET_PATH))
    print(f"  임베딩 shape : {X.shape}")
    print(f"  데이터셋 행수: {len(df)}")

    if X.shape[0] != len(df):
        raise ValueError(f"임베딩({X.shape[0]})과 데이터셋({len(df)}) 행 수가 다릅니다!")

    # 2. 모델 로드
    print("\n[2] 모델 로드 중...")
    domain_clf = joblib.load(str(DOMAIN_CLF_PATH))
    le_domain  = joblib.load(str(DOMAIN_LE_PATH))
    intent_clf = joblib.load(str(INTENT_CLF_PATH))
    le_intent  = joblib.load(str(INTENT_LE_PATH))
    print("  모델 로드 완료 ✅")

    # 3. 임베딩 품질 검증
    print("\n[3] 임베딩 품질 검증 중...")
    validate_embeddings(X, df)

    # 4. 도메인 분류기 평가
    print("\n[4] 도메인 분류기 평가 중...")
    y_domain = le_domain.transform(df["domain"])

    evaluate_classifier(
        clf          = domain_clf,
        X            = X,
        y_enc        = y_domain,
        label_names  = le_domain.classes_.tolist(),
        title        = "Domain Classifier",
        fig_filename = "domain_confusion_matrix.png",
    )

    plot_roc_curve(
        clf          = domain_clf,
        X            = X,
        y_enc        = y_domain,
        label_names  = le_domain.classes_.tolist(),
        title        = "Domain Classifier",
        fig_filename = "domain_roc.png",
    )

    # 5. 인텐트 분류기 평가 (도메인별)
    print("\n[5] 인텐트 분류기 평가 중...")
    for domain in le_domain.classes_:
        if domain not in intent_clf:
            print(f"  [{domain}] 인텐트 분류기 없음 — 스킵")
            continue

        mask         = df["domain"] == domain
        X_domain     = X[mask]
        y_intent     = le_intent[domain].transform(df.loc[mask, "intent"])
        intent_names = le_intent[domain].classes_.tolist()

        if len(y_intent) < LR_KFOLD:
            print(f"  [{domain}] 샘플 수 부족 ({len(y_intent)}건) — 스킵")
            continue

        # 파일 저장용 이름만 변경
        safe_domain = domain.replace("/", "_")

        print(f"\n  ── {domain} 인텐트 평가 ──")

        evaluate_classifier(
            clf          = intent_clf[domain],
            X            = X_domain,
            y_enc        = y_intent,
            label_names  = intent_names,
            title        = f"Intent Classifier - {domain}",
            fig_filename = f"intent_cm_{safe_domain}.png",
            cmap         = "Greens",
        )

        plot_roc_curve(
            clf          = intent_clf[domain],
            X            = X_domain,
            y_enc        = y_intent,
            label_names  = intent_names,
            title        = f"Intent Classifier - {domain}",
            fig_filename = f"intent_roc_{safe_domain}.png",
        )

    print("\n평가 완료! outputs/figures/ 폴더를 확인하세요.")

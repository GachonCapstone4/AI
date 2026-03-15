# ============================================================
# 데이터 로드 / Contrastive Pair 생성 / 저장 / 복원
# ============================================================

import os
import random
import pandas as pd
import numpy as np
from itertools import combinations
from sentence_transformers import InputExample

from config import (
    DATASET_PATH, PAIRS_CSV_PATH,
    MAX_POSITIVES_PER_INTENT, MAX_NEGATIVES_PER_SAMPLE,
    RANDOM_SEED
)

random.seed(RANDOM_SEED)


# ── 데이터 로드 및 검증 ─────────────────────────────────────
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """dataset.csv 로드 및 필수 컬럼 검증 / 결측값 제거"""
    df = pd.read_csv(path)
    required = ["email_text", "domain", "intent"]
    missing  = set(required) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"[load_dataset] 총 샘플: {len(df)} | 도메인: {df['domain'].nunique()} | 인텐트: {df['intent'].nunique()}")
    return df


# ── Contrastive Pair 생성 ───────────────────────────────────
def generate_contrastive_pairs(df: pd.DataFrame) -> list:
    """
    Positive  : 같은 intent
    Hard Neg  : 같은 domain, 다른 intent
    return    : List[InputExample]
    """
    pairs = []

    # Positive pairs
    for intent, group in df.groupby("intent"):
        indices = group.index.tolist()
        if len(indices) < 2:
            continue
        all_pos = list(combinations(indices, 2))
        sampled = random.sample(all_pos, min(MAX_POSITIVES_PER_INTENT, len(all_pos)))
        for i, j in sampled:
            pairs.append(InputExample(
                texts=[df.loc[i, "email_text"], df.loc[j, "email_text"]],
                label=1.0
            ))

    # Hard Negative pairs (같은 domain, 다른 intent)
    for domain, domain_df in df.groupby("domain"):
        if domain_df["intent"].nunique() < 2:
            continue
        for idx in domain_df.index:
            neg_pool = domain_df[
                domain_df["intent"] != df.loc[idx, "intent"]
            ].index.tolist()
            if not neg_pool:
                continue
            for j in random.sample(neg_pool, min(MAX_NEGATIVES_PER_SAMPLE, len(neg_pool))):
                pairs.append(InputExample(
                    texts=[df.loc[idx, "email_text"], df.loc[j, "email_text"]],
                    label=0.0
                ))

    random.shuffle(pairs)
    pos = sum(1 for p in pairs if p.label == 1.0)
    print(f"[generate_pairs] 총: {len(pairs)} | Positive: {pos} | Negative: {len(pairs)-pos}")
    return pairs


# ── Pair CSV 저장 ───────────────────────────────────────────
def save_pairs_csv(pairs: list, path: str = PAIRS_CSV_PATH) -> None:
    """InputExample 리스트 → CSV 저장"""
    pd.DataFrame([
        {"text_a": p.texts[0], "text_b": p.texts[1], "label": int(p.label)}
        for p in pairs
    ]).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[save_pairs_csv] 저장 완료 → {path}")


# ── Pair CSV 로드 → InputExample 복원 ──────────────────────
def load_pairs_csv(path: str = PAIRS_CSV_PATH) -> list:
    """CSV → InputExample 리스트 복원"""
    df_pairs = pd.read_csv(path, encoding="utf-8-sig")
    df_pairs["label"] = df_pairs["label"].astype(float)
    pairs = [
        InputExample(
            texts=[row["text_a"], row["text_b"]],
            label=row["label"]
        )
        for _, row in df_pairs.iterrows()
    ]
    print(f"[load_pairs_csv] 복원 완료: {len(pairs)}개")
    return pairs


# ── train / validation split ────────────────────────────────
def split_pairs(
    pairs    : list,
    val_ratio: float = 0.1,
    seed     : int   = RANDOM_SEED
) -> tuple:
    """
    return: (train_examples, val_examples)
    """
    random.seed(seed)
    val_size       = int(len(pairs) * val_ratio)
    val_indices    = set(random.sample(range(len(pairs)), val_size))
    train_examples = [p for i, p in enumerate(pairs) if i not in val_indices]
    val_examples   = [p for i, p in enumerate(pairs) if i in val_indices]
    print(f"[split_pairs] Train: {len(train_examples)} | Val: {len(val_examples)}")
    return train_examples, val_examples


# ── 임베딩 저장 / 로드 ──────────────────────────────────────
def save_embeddings(X: np.ndarray, path: str) -> None:
    """임베딩 numpy 배열 저장"""
    np.save(path, X)
    print(f"[save_embeddings] 저장 완료 → {path}")


def load_embeddings(path: str) -> np.ndarray:
    """임베딩 numpy 배열 로드"""
    X = np.load(path)
    print(f"[load_embeddings] 로드 완료 — shape: {X.shape}")
    return X
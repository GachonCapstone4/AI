# ============================================================
# SBERT Fine-tuning + 임베딩 생성
# ============================================================

import math
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, evaluation

from config import (
    SBERT_BASE_MODEL, SBERT_MODEL_PATH,
    SBERT_BATCH_SIZE, SBERT_EPOCHS,
    SBERT_WARMUP_RATIO, SBERT_VAL_RATIO,
    EMBEDDINGS_FINETUNED_PATH,
)
from data_utils import load_pairs_csv, split_pairs, save_embeddings


def build_evaluator(val_examples: list) -> evaluation.EmbeddingSimilarityEvaluator:
    """validation pair → EmbeddingSimilarityEvaluator 생성"""
    return evaluation.EmbeddingSimilarityEvaluator(
        sentences1=[e.texts[0] for e in val_examples],
        sentences2=[e.texts[1] for e in val_examples],
        scores    =[e.label     for e in val_examples],
        name      ="val_contrastive",
    )


def run_sbert_finetuning(
    output_path  : str   = SBERT_MODEL_PATH,
    base_model   : str   = SBERT_BASE_MODEL,
    batch_size   : int   = SBERT_BATCH_SIZE,
    epochs       : int   = SBERT_EPOCHS,
    warmup_ratio : float = SBERT_WARMUP_RATIO,
    val_ratio    : float = SBERT_VAL_RATIO,
    pairs         : list | None = None,
    pairs_csv_path: str | None = None,
) -> SentenceTransformer:
    """
    pair CSV 로드 → train/val split → ContrastiveLoss fine-tuning
    return: fine-tuned SentenceTransformer
    """
    if pairs is None:
        pairs = load_pairs_csv(pairs_csv_path) if pairs_csv_path else load_pairs_csv()
    train_examples, val_examples = split_pairs(pairs, val_ratio)

    model            = SentenceTransformer(base_model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss       = losses.ContrastiveLoss(model)
    evaluator        = build_evaluator(val_examples)
    warmup_steps     = math.ceil(len(train_dataloader) * epochs * warmup_ratio)

    print(f"[run_sbert_finetuning] 배치: {len(train_dataloader)} | warmup: {warmup_steps}")
    print(f"저장 경로: {output_path}\n")

    model.fit(
        train_objectives  =[(train_dataloader, train_loss)],
        evaluator         =evaluator,
        evaluation_steps  =len(train_dataloader),
        epochs            =epochs,
        warmup_steps      =warmup_steps,
        output_path       =output_path,
        show_progress_bar =True,
        save_best_model   =True,
    )

    print(f"[run_sbert_finetuning] 완료 → {output_path}")
    return SentenceTransformer(output_path)


def generate_embeddings(
    texts      : list,
    model_path : str = SBERT_MODEL_PATH,
    save_path  : str = EMBEDDINGS_FINETUNED_PATH,
    batch_size : int = 64,
) -> np.ndarray:
    """
    fine-tuned SBERT → 정규화 임베딩 생성 + data/ 에 저장
    return: np.ndarray (N, 384)
    """
    model = SentenceTransformer(model_path)
    X = model.encode(
        texts,
        batch_size          =batch_size,
        show_progress_bar   =True,
        normalize_embeddings=True,
    )
    print(f"[generate_embeddings] shape: {X.shape}")
    if save_path:
        save_embeddings(X, save_path)
    return X

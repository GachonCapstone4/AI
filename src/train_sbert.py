# ============================================================
# SBERT Fine-tuning + 임베딩 생성
# ============================================================

import math
import shutil
from pathlib import Path

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


SBERT_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
SBERT_ROOT_REQUIRED_FILES = ("modules.json", "tokenizer.json")
SBERT_CONFIG_CANDIDATES = (
    "config.json",
    "config_sentence_transformers.json",
    "0_Transformer/config.json",
)


def _log_directory_tree(root: str | Path, label: str) -> None:
    root_path = Path(root)
    print(f"[{label}] directory tree: {root_path}", flush=True)
    if not root_path.exists():
        print(f"[{label}] MISSING: {root_path}", flush=True)
        return

    for path in sorted(root_path.rglob("*")):
        relative = path.relative_to(root_path).as_posix()
        suffix = "/" if path.is_dir() else f" ({path.stat().st_size} bytes)"
        print(f"[{label}]   {relative}{suffix}", flush=True)


def _copy_transformer_core_files_to_root(output_dir: Path) -> None:
    """Expose core transformer files at the SBERT root for preload validators."""
    transformer_dir = output_dir / "0_Transformer"
    if not transformer_dir.is_dir():
        return

    for filename in (*SBERT_WEIGHT_FILES, "tokenizer.json", "config.json"):
        source = transformer_dir / filename
        destination = output_dir / filename
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)


def _missing_sbert_artifact_paths(output_dir: str | Path) -> list[str]:
    output_path = Path(output_dir)
    missing: list[str] = []

    if not output_path.is_dir():
        return [f"{output_path} (expected directory)"]

    if not any((output_path / filename).is_file() for filename in SBERT_WEIGHT_FILES):
        missing.append("model.safetensors or pytorch_model.bin")

    for filename in SBERT_ROOT_REQUIRED_FILES:
        if not (output_path / filename).is_file():
            missing.append(filename)

    if not any((output_path / filename).is_file() for filename in SBERT_CONFIG_CANDIDATES):
        missing.append("config.json or config_sentence_transformers.json")

    if not (output_path / "1_Pooling").is_dir():
        missing.append("1_Pooling/")

    return missing


def _finalize_and_reload_sbert(model: SentenceTransformer, output_path: str | Path) -> SentenceTransformer:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log_directory_tree(output_dir, "run_sbert_finetuning.after_fit")

    # model.fit(save_best_model=True) should write the best evaluated model to
    # output_path. Reload it first when possible, then save again to normalize
    # the final root artifact. If the best model was not materialized, fall back
    # to the in-memory final model instead of leaving only checkpoint files.
    try:
        model_to_save = SentenceTransformer(str(output_dir))
        print(f"[run_sbert_finetuning] best model reload succeeded: {output_dir}", flush=True)
    except Exception as exc:
        print(
            "[run_sbert_finetuning] best model reload failed before final save; "
            f"saving in-memory model to output_path. error={exc}",
            flush=True,
        )
        model_to_save = model

    model_to_save.save(str(output_dir))
    _copy_transformer_core_files_to_root(output_dir)

    missing = _missing_sbert_artifact_paths(output_dir)
    _log_directory_tree(output_dir, "run_sbert_finetuning.after_save")
    if missing:
        raise FileNotFoundError(
            "Incomplete SBERT artifact at "
            f"{output_dir}. Missing required paths: {', '.join(missing)}"
        )

    try:
        reloaded = SentenceTransformer(str(output_dir))
    except Exception as exc:
        raise RuntimeError(f"Failed to reload finalized SBERT artifact from {output_dir}: {exc}") from exc

    print(f"[run_sbert_finetuning] reload validation succeeded: {output_dir}", flush=True)
    return reloaded


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

    output_dir = Path(output_path)
    print(f"[run_sbert_finetuning] 배치: {len(train_dataloader)} | warmup: {warmup_steps}")
    print(f"[run_sbert_finetuning] output_path: {output_dir.resolve()}\n")

    model.fit(
        train_objectives  =[(train_dataloader, train_loss)],
        evaluator         =evaluator,
        evaluation_steps  =len(train_dataloader),
        epochs            =epochs,
        warmup_steps      =warmup_steps,
        output_path       =str(output_dir),
        show_progress_bar =True,
        save_best_model   =True,
    )

    print(f"[run_sbert_finetuning] fit completed: {output_dir}")
    return _finalize_and_reload_sbert(model, output_dir)


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

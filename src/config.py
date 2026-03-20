# ============================================================
# 전역 경로 및 하이퍼파라미터 설정
# ============================================================

import os
from pathlib import Path

# ── 루트 경로 ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── 5개 하위 폴더 ───────────────────────────────────────────
DATA_DIR      = BASE_DIR / "data"
MODEL_DIR     = BASE_DIR / "models"
SRC_DIR       = BASE_DIR / "src"
NOTEBOOK_DIR  = BASE_DIR / "notebooks"
OUTPUT_DIR    = BASE_DIR / "outputs"

# ── outputs 하위 폴더 ───────────────────────────────────────
FIGURES_DIR   = OUTPUT_DIR / "figures"
REPORTS_DIR   = OUTPUT_DIR / "reports"
LOG_DIR       = OUTPUT_DIR / "logs"

# ── 데이터 파일 경로 ────────────────────────────────────────
DATASET_PATH              = DATA_DIR / "dataset_new.csv"
PAIRS_CSV_PATH            = DATA_DIR / "contrastive_pairs.csv"
EMBEDDINGS_BASELINE_PATH  = DATA_DIR / "embeddings_baseline.npy"
EMBEDDINGS_FINETUNED_PATH = DATA_DIR / "embeddings_finetuned.npy"

# ── 모델 저장 경로 ──────────────────────────────────────────
SBERT_MODEL_PATH  = MODEL_DIR / "sbert_business_email"
DOMAIN_CLF_PATH   = MODEL_DIR / "domain_classifier.pkl"
DOMAIN_LE_PATH    = MODEL_DIR / "domain_label_encoder.pkl"
INTENT_CLF_PATH   = MODEL_DIR / "intent_classifiers.pkl"
INTENT_LE_PATH    = MODEL_DIR / "intent_label_encoders.pkl"

# ── outputs 파일 경로 ───────────────────────────────────────
DOMAIN_CM_PATH    = FIGURES_DIR / "domain_confusion_matrix.png"

# ── SBERT 하이퍼파라미터 ────────────────────────────────────
SBERT_BASE_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SBERT_BATCH_SIZE   = 16
SBERT_EPOCHS       = 5
SBERT_WARMUP_RATIO = 0.1
SBERT_VAL_RATIO    = 0.1

# ── Pair 생성 파라미터 ──────────────────────────────────────
MAX_POSITIVES_PER_INTENT = 30
MAX_NEGATIVES_PER_SAMPLE = 3
RANDOM_SEED              = 42

# ── 분류기 파라미터 ─────────────────────────────────────────
LR_MAX_ITER  = 1000
LR_C         = 1.0
LR_SOLVER    = "lbfgs"
LR_KFOLD     = 5

# ── 추론 파라미터 ───────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4

# ── 디렉터리 자동 생성 ──────────────────────────────────────
for _dir in [DATA_DIR, MODEL_DIR, SRC_DIR, NOTEBOOK_DIR,
             FIGURES_DIR, REPORTS_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
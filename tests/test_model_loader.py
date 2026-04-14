from pathlib import Path
import sys
import types

import pytest

sys.modules.setdefault("joblib", types.SimpleNamespace(load=lambda *_args, **_kwargs: None))
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=object),
)

from src.model_loader import RuntimeModelPaths, _validate_required_local_paths


def _make_runtime_paths(root: Path) -> RuntimeModelPaths:
    return RuntimeModelPaths(
        sbert_dir=root / "sbert",
        sbert_model_path=root / "sbert" / "model.safetensors",
        sbert_tokenizer_path=root / "sbert" / "tokenizer.json",
        domain_clf_path=root / "domain_classifier.pkl",
        domain_le_path=root / "domain_label_encoder.pkl",
        intent_clf_path=root / "intent_classifiers.pkl",
        intent_le_path=root / "intent_label_encoders.pkl",
    )


def test_validate_required_local_paths_accepts_required_runtime_files(tmp_path: Path):
    paths = _make_runtime_paths(tmp_path)

    paths.sbert_dir.mkdir(parents=True)
    paths.sbert_model_path.write_text("ok", encoding="utf-8")
    paths.sbert_tokenizer_path.write_text("ok", encoding="utf-8")
    paths.domain_clf_path.write_text("ok", encoding="utf-8")
    paths.domain_le_path.write_text("ok", encoding="utf-8")
    paths.intent_clf_path.write_text("ok", encoding="utf-8")
    paths.intent_le_path.write_text("ok", encoding="utf-8")
    (tmp_path / "README.md").write_text("not required", encoding="utf-8")

    _validate_required_local_paths(paths)


def test_validate_required_local_paths_fails_when_sbert_required_file_missing(tmp_path: Path):
    paths = _make_runtime_paths(tmp_path)

    paths.sbert_dir.mkdir(parents=True)
    paths.sbert_model_path.write_text("ok", encoding="utf-8")
    paths.domain_clf_path.write_text("ok", encoding="utf-8")
    paths.domain_le_path.write_text("ok", encoding="utf-8")
    paths.intent_clf_path.write_text("ok", encoding="utf-8")
    paths.intent_le_path.write_text("ok", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc:
        _validate_required_local_paths(paths)

    assert "sbert/tokenizer.json" in str(exc.value)

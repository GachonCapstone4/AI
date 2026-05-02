from pathlib import Path
import sys
import types

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.modules.setdefault("joblib", types.SimpleNamespace(load=lambda *_args, **_kwargs: None))
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=object),
)

import src.model_loader as model_loader
from src.model_loader import (
    RuntimeModelPaths,
    ensure_standard_model_artifact_cached,
    _validate_required_local_paths,
    parse_latest_model_version,
)


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


def test_parse_latest_model_version_accepts_snake_case():
    assert parse_latest_model_version({"model_version": "training-final-004"}) == "training-final-004"


def test_parse_latest_model_version_accepts_camel_case():
    assert parse_latest_model_version({"modelVersion": "training-final-005"}) == "training-final-005"


def test_parse_latest_model_version_fails_on_missing_version():
    with pytest.raises(RuntimeError, match="model_version"):
        parse_latest_model_version({"updated_at": "2026-04-30T13:35:13Z"})


def test_parse_latest_model_version_fails_when_only_artifact_s3_uri_exists():
    with pytest.raises(RuntimeError, match="model_version"):
        parse_latest_model_version(
            {
                "artifact_s3_uri": "s3://capstone-gachon/models/training-final-004/",
            }
        )


def test_parse_latest_model_version_fails_on_conflicting_keys():
    with pytest.raises(RuntimeError, match="conflicting"):
        parse_latest_model_version(
            {
                "model_version": "training-final-004",
                "modelVersion": "training-final-005",
            }
        )


def _write_standard_artifact(root: Path) -> None:
    (root / "sbert").mkdir(parents=True)
    (root / "sbert" / "model.safetensors").write_text("ok", encoding="utf-8")
    (root / "sbert" / "tokenizer.json").write_text("ok", encoding="utf-8")
    (root / "domain_model.pkl").write_text("ok", encoding="utf-8")
    (root / "intent_model.pkl").write_text("ok", encoding="utf-8")
    (root / "label_mapping.json").write_text("{}", encoding="utf-8")
    (root / "metrics.json").write_text("{}", encoding="utf-8")
    (root / "config.json").write_text("{}", encoding="utf-8")


def test_existing_complete_standard_cache_is_reused(tmp_path: Path, monkeypatch):
    artifact_dir = tmp_path / "v-complete"
    _write_standard_artifact(artifact_dir)
    (artifact_dir / ".complete").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(
        model_loader,
        "get_settings",
        lambda: types.SimpleNamespace(
            MODEL_SOURCE="s3",
            MODEL_LOCAL_CACHE_DIR=str(tmp_path),
            AWS_REGION="ap-northeast-2",
            S3_MODEL_BUCKET="bucket",
            S3_MODEL_PREFIX="models",
        ),
    )

    class FailCache:
        def __init__(self, **_kwargs):
            raise AssertionError("complete cache should not download")

    monkeypatch.setattr(model_loader, "S3ArtifactCache", FailCache)

    assert ensure_standard_model_artifact_cached("v-complete") == artifact_dir


def test_failed_s3_cache_download_cleans_temp_and_does_not_mark_complete(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        model_loader,
        "get_settings",
        lambda: types.SimpleNamespace(
            MODEL_SOURCE="s3",
            MODEL_LOCAL_CACHE_DIR=str(tmp_path),
            AWS_REGION="ap-northeast-2",
            S3_MODEL_BUCKET="bucket",
            S3_MODEL_PREFIX="models",
        ),
    )

    class IncompleteCache:
        def __init__(self, **_kwargs):
            pass

        def download_prefix(self, *, bucket, prefix, target_dir):
            (target_dir / "sbert").mkdir(parents=True)
            (target_dir / "sbert" / "model.safetensors").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(model_loader, "S3ArtifactCache", IncompleteCache)

    with pytest.raises(RuntimeError, match="Standard model artifacts are missing"):
        ensure_standard_model_artifact_cached("v-broken")

    assert not (tmp_path / "v-broken").exists()
    assert not list(tmp_path.glob("v-broken.tmp-*"))

import sys
import types
from pathlib import Path

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

import model_manager
from model_manager import ModelManager


def _settings(**overrides):
    values = {
        "MODEL_SOURCE": "s3",
        "ACTIVE_MODEL_VERSION": None,
        "S3_MODEL_BUCKET": "bucket",
        "S3_MODEL_PREFIX": "models",
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _valid_bundle(name: str = "bundle") -> dict:
    return {
        "name": name,
        "label_mapping": {
            "domains": ["Finance"],
            "intents": {"Finance": ["Invoice"]},
        },
        "runtime": {"active_model_version": name},
    }


def test_s3_startup_uses_latest_or_active_override_only(monkeypatch):
    manager = ModelManager()

    monkeypatch.setattr(model_manager, "get_settings", lambda: _settings())
    monkeypatch.setattr(model_manager, "load_latest_model_version", lambda: "latest-v1")
    monkeypatch.setattr(
        model_manager,
        "load_standard_model_bundle",
        lambda version: _valid_bundle(version),
    )

    def fail_legacy_loader():
        raise AssertionError("legacy loader must not be called for MODEL_SOURCE=s3")

    monkeypatch.setattr(model_manager, "load_classification_pipeline", fail_legacy_loader)

    result = manager.load_initial_model()

    assert result["model_version"] == "latest-v1"
    assert manager.current_bundle["name"] == "latest-v1"


def test_preload_does_not_change_current_bundle(monkeypatch):
    manager = ModelManager()
    current = _valid_bundle("current")
    manager.current_bundle = current
    manager.current_model_version = "current"

    monkeypatch.setattr(model_manager, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        model_manager,
        "load_standard_model_bundle",
        lambda version: _valid_bundle(version),
    )

    result = manager.preload("staging")

    assert result["status"] == "preloaded"
    assert manager.current_bundle is current
    assert manager.current_model_version == "current"
    assert manager.staging_bundle["name"] == "staging"
    assert manager.staging_validated is False


def test_validate_failure_makes_switch_impossible(monkeypatch):
    manager = ModelManager()
    manager.current_bundle = _valid_bundle("current")
    manager.current_model_version = "current"
    manager.staging_bundle = _valid_bundle("staging")
    manager.staging_model_version = "staging"

    def fake_predict_email(_text, _bundle):
        return {"domain": "Finance", "intent": "NotInMapping", "confidence_score": 0.5}

    monkeypatch.setattr(model_manager, "predict_email", fake_predict_email)

    with pytest.raises(RuntimeError, match="Predicted intent"):
        manager.validate()
    with pytest.raises(RuntimeError, match="has not been validated"):
        manager.switch()

    assert manager.current_model_version == "current"
    assert manager.current_bundle["name"] == "current"


def test_validate_success_then_switch_replaces_current_and_clears_staging(monkeypatch):
    manager = ModelManager()
    current = _valid_bundle("current")
    staging = _valid_bundle("staging")
    manager.current_bundle = current
    manager.current_model_version = "current"
    manager.staging_bundle = staging
    manager.staging_model_version = "staging"

    def fake_predict_email(_text, bundle):
        assert bundle is staging
        return {"domain": "Finance", "intent": "Invoice", "confidence_score": 0.99}

    monkeypatch.setattr(model_manager, "predict_email", fake_predict_email)

    validate_result = manager.validate()
    switch_result = manager.switch()

    assert validate_result["status"] == "validated"
    assert switch_result == {"status": "switched", "model_version": "staging"}
    assert manager.current_bundle is staging
    assert manager.current_model_version == "staging"
    assert manager.staging_bundle is None
    assert manager.staging_model_version is None
    assert manager.staging_validated is False


def test_preload_after_validate_resets_validation_state(monkeypatch):
    manager = ModelManager()
    first = _valid_bundle("first")
    manager.staging_bundle = first
    manager.staging_model_version = "first"

    monkeypatch.setattr(
        model_manager,
        "predict_email",
        lambda _text, _bundle: {
            "domain": "Finance",
            "intent": "Invoice",
            "confidence_score": 0.99,
        },
    )
    manager.validate()
    assert manager.staging_validated is True

    monkeypatch.setattr(model_manager, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        model_manager,
        "load_standard_model_bundle",
        lambda version: _valid_bundle(version),
    )

    manager.preload("second")

    assert manager.staging_model_version == "second"
    assert manager.staging_bundle["name"] == "second"
    assert manager.staging_validated is False


def test_latest_json_based_preload_works_without_model_version(monkeypatch):
    manager = ModelManager()

    monkeypatch.setattr(model_manager, "get_settings", lambda: _settings())
    monkeypatch.setattr(model_manager, "load_latest_model_version", lambda: "latest-v2")
    monkeypatch.setattr(
        model_manager,
        "load_standard_model_bundle",
        lambda version: _valid_bundle(version),
    )

    result = manager.preload()

    assert result["status"] == "preloaded"
    assert result["model_version"] == "latest-v2"
    assert manager.staging_model_version == "latest-v2"


def test_preload_failure_does_not_pollute_current_or_staging(monkeypatch):
    manager = ModelManager()
    current = _valid_bundle("current")
    staging = _valid_bundle("existing-staging")
    manager.current_bundle = current
    manager.current_model_version = "current"
    manager.staging_bundle = staging
    manager.staging_model_version = "existing-staging"
    manager.staging_validated = True

    monkeypatch.setattr(model_manager, "get_settings", lambda: _settings())

    def raise_missing_artifact(_version):
        raise RuntimeError("Standard model artifacts are missing: domain_model.pkl")

    monkeypatch.setattr(model_manager, "load_standard_model_bundle", raise_missing_artifact)

    with pytest.raises(RuntimeError, match="domain_model.pkl"):
        manager.preload("broken")

    assert manager.current_bundle is current
    assert manager.current_model_version == "current"
    assert manager.staging_bundle is staging
    assert manager.staging_model_version == "existing-staging"
    assert manager.staging_validated is False


def test_preload_failure_after_validated_staging_requires_revalidation(monkeypatch):
    manager = ModelManager()
    current = _valid_bundle("current")
    staging = _valid_bundle("validated-staging")
    manager.current_bundle = current
    manager.current_model_version = "current"
    manager.staging_bundle = staging
    manager.staging_model_version = "validated-staging"
    manager.staging_validated = True

    monkeypatch.setattr(model_manager, "get_settings", lambda: _settings())

    def raise_missing_artifact(_version):
        raise RuntimeError("Standard model artifacts are missing: intent_model.pkl")

    monkeypatch.setattr(model_manager, "load_standard_model_bundle", raise_missing_artifact)

    with pytest.raises(RuntimeError, match="intent_model.pkl"):
        manager.preload("broken")
    with pytest.raises(RuntimeError, match="has not been validated"):
        manager.switch()

    assert manager.current_bundle is current
    assert manager.current_model_version == "current"
    assert manager.staging_bundle is staging
    assert manager.staging_model_version == "validated-staging"
    assert manager.staging_validated is False


def test_validate_uses_staging_bundle_and_label_mapping(monkeypatch):
    manager = ModelManager()
    manager.staging_model_version = "v-test"
    manager.staging_bundle = {
        "label_mapping": {
            "domains": ["Finance"],
            "intents": {"Finance": ["Invoice"]},
        }
    }

    def fake_predict_email(_text, bundle):
        assert bundle is manager.staging_bundle
        return {
            "domain": "Finance",
            "intent": "Invoice",
            "confidence_score": 0.99,
        }

    monkeypatch.setattr(model_manager, "predict_email", fake_predict_email)

    result = manager.validate()

    assert result["status"] == "validated"
    assert result["model_version"] == "v-test"
    assert len(result["samples"]) == 2


def test_switch_replaces_current_bundle_by_reference():
    manager = ModelManager()
    old_bundle = {"name": "old"}
    new_bundle = {"name": "new"}
    manager.current_bundle = old_bundle
    manager.current_model_version = "v-old"
    manager.staging_bundle = new_bundle
    manager.staging_model_version = "v-new"
    manager.staging_validated = True

    result = manager.switch()

    assert result == {"status": "switched", "model_version": "v-new"}
    assert manager.current_bundle is new_bundle
    assert manager.current_model_version == "v-new"
    assert manager.staging_bundle is None
    assert manager.staging_model_version is None
    assert manager.staging_validated is False


def test_predict_copies_current_bundle_reference_before_prediction(monkeypatch):
    manager = ModelManager()
    original_bundle = {"name": "current"}
    replacement_bundle = {"name": "replacement"}
    manager.current_bundle = original_bundle

    def fake_predict_email(_text, bundle):
        manager.current_bundle = replacement_bundle
        assert bundle is original_bundle
        return {"domain": "Finance", "intent": "Invoice"}

    monkeypatch.setattr(model_manager, "predict_email", fake_predict_email)

    result = manager.predict("hello")

    assert result == {"domain": "Finance", "intent": "Invoice"}
    assert manager.current_bundle is replacement_bundle

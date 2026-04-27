import sys
import types
from pathlib import Path


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

    assert result["status"] == "ok"
    assert result["model_version"] == "v-test"
    assert len(result["predictions"]) == 2


def test_switch_replaces_current_bundle_by_reference():
    manager = ModelManager()
    old_bundle = {"name": "old"}
    new_bundle = {"name": "new"}
    manager.current_bundle = old_bundle
    manager.current_model_version = "v-old"
    manager.staging_bundle = new_bundle
    manager.staging_model_version = "v-new"

    result = manager.switch()

    assert result == {"status": "ok", "model_version": "v-new"}
    assert manager.current_bundle is new_bundle
    assert manager.current_model_version == "v-new"
    assert manager.staging_bundle is None
    assert manager.staging_model_version is None


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

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CALLS = {
    "cross_val_score": 0,
    "cross_val_predict": 0,
    "fit": 0,
}


def _install_training_stubs() -> None:
    sklearn = types.ModuleType("sklearn")
    model_selection = types.ModuleType("sklearn.model_selection")
    metrics = types.ModuleType("sklearn.metrics")
    preprocessing = types.ModuleType("sklearn.preprocessing")
    linear_model = types.ModuleType("sklearn.linear_model")
    pairwise = types.ModuleType("sklearn.metrics.pairwise")

    class StratifiedKFold:
        def __init__(self, n_splits=5, shuffle=False, random_state=None) -> None:
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

    def cross_val_score(_clf, _X, _y, cv, scoring):
        CALLS["cross_val_score"] += 1
        return np.array([0.75] * cv.n_splits)

    def cross_val_predict(_clf, _X, y, cv, method=None):
        CALLS["cross_val_predict"] += 1
        if method == "predict_proba":
            return np.ones((len(y), len(np.unique(y)))) / len(np.unique(y))
        return np.asarray(y)

    def classification_report(*_args, **_kwargs):
        return "classification report"

    def confusion_matrix(y_true, y_pred):
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
        index = {label: idx for idx, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for true, pred in zip(y_true, y_pred):
            matrix[index[true], index[pred]] += 1
        return matrix

    def f1_score(*_args, **_kwargs):
        return 1.0

    def accuracy_score(y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    def roc_curve(*_args, **_kwargs):
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5])

    def auc(_x, _y):
        return 1.0

    def label_binarize(y, classes):
        result = np.zeros((len(y), len(classes)), dtype=int)
        for row, value in enumerate(y):
            result[row, classes.index(value)] = 1
        return result

    def cosine_similarity(X):
        return np.eye(len(X))

    class LabelEncoder:
        def fit_transform(self, y):
            self.classes_ = np.array(sorted(set(y)))
            return self.transform(y)

        def transform(self, y):
            index = {label: idx for idx, label in enumerate(self.classes_)}
            return np.array([index[value] for value in y])

        def inverse_transform(self, y):
            return np.array([self.classes_[int(value)] for value in y])

    class LogisticRegression:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def fit(self, _X, y):
            CALLS["fit"] += 1
            if len(np.unique(y)) < 2:
                raise ValueError(
                    "This solver needs samples of at least 2 classes in the data"
                )
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    model_selection.StratifiedKFold = StratifiedKFold
    model_selection.cross_val_score = cross_val_score
    model_selection.cross_val_predict = cross_val_predict
    metrics.classification_report = classification_report
    metrics.confusion_matrix = confusion_matrix
    metrics.roc_curve = roc_curve
    metrics.auc = auc
    metrics.f1_score = f1_score
    metrics.accuracy_score = accuracy_score
    preprocessing.label_binarize = label_binarize
    preprocessing.LabelEncoder = LabelEncoder
    linear_model.LogisticRegression = LogisticRegression
    pairwise.cosine_similarity = cosine_similarity

    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.model_selection"] = model_selection
    sys.modules["sklearn.metrics"] = metrics
    sys.modules["sklearn.preprocessing"] = preprocessing
    sys.modules["sklearn.linear_model"] = linear_model
    sys.modules["sklearn.metrics.pairwise"] = pairwise

    if "joblib" not in sys.modules:
        joblib = types.ModuleType("joblib")
        joblib.dump = lambda *_args, **_kwargs: None
        sys.modules["joblib"] = joblib

    if "seaborn" not in sys.modules and importlib.util.find_spec("seaborn") is None:
        seaborn = types.ModuleType("seaborn")
        seaborn.set_theme = lambda *args, **kwargs: None
        seaborn.heatmap = lambda *args, **kwargs: None
        sys.modules["seaborn"] = seaborn


def _reload_training_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_training_stubs()
    for name in ("src.evaluation", "evaluation", "src.train_domain", "src.train_intent"):
        sys.modules.pop(name, None)

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    monkeypatch.syspath_prepend(str(src))
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    evaluation = importlib.import_module("src.evaluation")
    monkeypatch.setattr(evaluation, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(evaluation.plt, "show", lambda: None)
    train_domain = importlib.import_module("src.train_domain")
    train_intent = importlib.import_module("src.train_intent")
    return evaluation, train_domain, train_intent


def _reset_calls() -> None:
    for key in CALLS:
        CALLS[key] = 0


def test_evaluate_classifier_skips_cv_when_min_class_count_is_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluation, _, _ = _reload_training_modules(monkeypatch, tmp_path)
    _reset_calls()

    result = evaluation.evaluate_classifier(
        clf=object(),
        X=np.zeros((5, 2)),
        y_enc=np.array([0, 0, 0, 0, 1]),
        label_names=["A", "B"],
        title="Domain Classifier",
        fig_filename="domain.png",
        n_splits=5,
    )

    assert result["cv_skipped"] is True
    assert result["cv_skip_reason"] == "min_class_count < 2"
    assert result["label_distribution"] == {"A": 4, "B": 1}
    assert CALLS["cross_val_predict"] == 0


def test_domain_training_fails_clearly_when_only_one_domain_class(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _, train_domain, _ = _reload_training_modules(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="requires at least 2 domain classes"):
        train_domain.train_domain_classifier(
            X=np.zeros((3, 2)),
            y_domain=np.array(["Only", "Only", "Only"]),
            model_path=tmp_path / "domain.pkl",
            label_encoder_path=None,
        )


def test_evaluate_classifier_keeps_kfold_for_normal_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evaluation, _, _ = _reload_training_modules(monkeypatch, tmp_path)
    _reset_calls()

    result = evaluation.evaluate_classifier(
        clf=object(),
        X=np.zeros((6, 2)),
        y_enc=np.array([0, 0, 0, 1, 1, 1]),
        label_names=["A", "B"],
        title="Domain Classifier",
        fig_filename="domain.png",
        n_splits=3,
    )

    assert result["cv_skipped"] is False
    assert result["effective_n_splits"] == 3
    assert result["weighted_f1_mean"] == 0.75
    assert CALLS["cross_val_score"] == 2
    assert CALLS["cross_val_predict"] == 1


def test_domain_training_returns_cv_metadata_for_rare_class(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _, train_domain, _ = _reload_training_modules(monkeypatch, tmp_path)

    _clf, _encoder, metadata = train_domain.train_domain_classifier(
        X=np.zeros((5, 2)),
        y_domain=np.array(["A", "A", "A", "A", "B"]),
        model_path=tmp_path / "domain.pkl",
        label_encoder_path=None,
        return_metadata=True,
    )

    assert metadata["domain_cv"]["cv_skipped"] is True
    assert metadata["domain_cv"]["cv_skip_reason"] == "min_class_count < 2"
    assert metadata["domain_label_distribution"] == {"A": 4, "B": 1}


def test_intent_training_skips_single_intent_domain_and_evaluates_valid_domain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _, _, train_intent = _reload_training_modules(monkeypatch, tmp_path)

    df = pd.DataFrame(
        {
            "domain": ["Sales", "Sales", "Finance", "Finance", "Finance", "Finance"],
            "intent": ["Quote", "Quote", "Invoice", "Invoice", "Tax", "Tax"],
        }
    )
    classifiers, encoders, metadata = train_intent.train_intent_classifiers(
        X=np.zeros((6, 2)),
        df=df,
        model_path=tmp_path / "intent.pkl",
        label_encoders_path=None,
        return_metadata=True,
    )

    assert "Sales" not in classifiers
    assert "Sales" not in encoders
    assert "requires at least 2 intent classes" in metadata["intent_training_skipped"]["Sales"]
    assert "Finance" in classifiers
    assert metadata["intent_cv"]["Finance"]["cv_skipped"] is False
    assert metadata["intent_label_distribution"]["Sales"] == {"Quote": 2}

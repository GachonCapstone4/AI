from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "sentence_transformers" not in sys.modules:
    sentence_transformers = types.ModuleType("sentence_transformers")

    class _InputExample:
        def __init__(self, texts, label) -> None:
            self.texts = texts
            self.label = label

    sentence_transformers.SentenceTransformer = object
    sentence_transformers.InputExample = _InputExample
    sentence_transformers.losses = SimpleNamespace(ContrastiveLoss=lambda _model: object())
    sentence_transformers.evaluation = SimpleNamespace(EmbeddingSimilarityEvaluator=lambda **_kwargs: object())
    sys.modules["sentence_transformers"] = sentence_transformers

import src.train_sbert as train_sbert


class _Pair:
    def __init__(self, text_a: str, text_b: str, label: float) -> None:
        self.texts = [text_a, text_b]
        self.label = label


class _FakeSentenceTransformer:
    constructed_from: list[str] = []
    fit_output_paths: list[str] = []
    save_paths: list[str] = []

    def __init__(self, source: str) -> None:
        self.source = str(source)
        self.constructed_from.append(self.source)
        path = Path(self.source)
        if path.exists() and not (path / "modules.json").is_file():
            raise OSError(
                "Error no file named pytorch_model.bin, model.safetensors, "
                "tf_model.h5, model.ckpt.index or flax_model.msgpack found"
            )

    def fit(self, **kwargs) -> None:
        output_path = Path(kwargs["output_path"])
        self.fit_output_paths.append(str(output_path))
        checkpoint = output_path / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"checkpoint")

    def save(self, output_path: str) -> None:
        root = Path(output_path)
        self.save_paths.append(str(root))
        transformer = root / "0_Transformer"
        pooling = root / "1_Pooling"
        transformer.mkdir(parents=True, exist_ok=True)
        pooling.mkdir(parents=True, exist_ok=True)
        (root / "modules.json").write_text("[]", encoding="utf-8")
        (root / "config_sentence_transformers.json").write_text("{}", encoding="utf-8")
        (transformer / "model.safetensors").write_bytes(b"model")
        (transformer / "tokenizer.json").write_text("{}", encoding="utf-8")
        (transformer / "config.json").write_text("{}", encoding="utf-8")
        (pooling / "config.json").write_text("{}", encoding="utf-8")


def test_run_sbert_finetuning_saves_final_reloadable_sbert_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "sbert"
    _FakeSentenceTransformer.constructed_from = []
    _FakeSentenceTransformer.fit_output_paths = []
    _FakeSentenceTransformer.save_paths = []

    monkeypatch.setattr(train_sbert, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(train_sbert.losses, "ContrastiveLoss", lambda _model: object())
    monkeypatch.setattr(
        train_sbert.evaluation,
        "EmbeddingSimilarityEvaluator",
        lambda **_kwargs: object(),
    )

    model = train_sbert.run_sbert_finetuning(
        output_path=str(output_path),
        base_model="fake-base",
        batch_size=2,
        epochs=1,
        val_ratio=0.5,
        pairs=[
            _Pair("domain invoice", "invoice question", 1.0),
            _Pair("domain refund", "shipping delay", 0.0),
        ],
    )

    assert isinstance(model, _FakeSentenceTransformer)
    assert _FakeSentenceTransformer.fit_output_paths == [str(output_path)]
    assert _FakeSentenceTransformer.save_paths == [str(output_path)]
    assert (output_path / "model.safetensors").is_file()
    assert (output_path / "tokenizer.json").is_file()
    assert (output_path / "modules.json").is_file()
    assert (output_path / "config.json").is_file()
    assert (output_path / "1_Pooling").is_dir()
    assert not (output_path / "sbert").exists()
    assert _FakeSentenceTransformer.constructed_from[-1] == str(output_path)

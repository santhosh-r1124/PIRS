"""Text embedders for the resolution recommender. Each returns L2-normalised vectors, so a
dot product is the cosine similarity.

  minilm       sentence-transformers all-MiniLM-L6-v2 (the spec's model). Loaded from
               recommender.model_dir if that folder exists (offline), else from the Hub.
  minilm_onnx  the same model as ChromaDB's ONNX build. Uses recommender.onnx_archive if
               present; the archive is checked against the SHA-256 that chromadb itself pins
               before it is used.
  tfidf        word + bigram TF-IDF (scikit-learn): a keyword-matching baseline, fit on the
               indexed (training-period) tickets only.

If a requested embedder is not available, EmbedderUnavailable is raised - there is no silent
fallback, so results are never labelled with the wrong model.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path

import numpy as np

from src.utils import ROOT


class EmbedderUnavailable(RuntimeError):
    pass


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, rcfg: dict):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            ngram_range=(1, 2), max_features=rcfg["tfidf_max_features"], sublinear_tf=True
        )

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        self.vec.fit(texts)
        return self

    def __call__(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.vec.transform(texts).toarray())


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, rcfg: dict):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbedderUnavailable(f"sentence-transformers not installed: {e}") from e
        local = ROOT / rcfg["model_dir"]
        try:
            self.model = SentenceTransformer(
                str(local) if local.exists() else rcfg["model_name"], device="cpu"
            )
        except Exception as e:  # no local copy and the Hub is unreachable
            raise EmbedderUnavailable(
                f"could not load {rcfg['model_name']}: {type(e).__name__}"
            ) from e
        self.source = str(local) if local.exists() else rcfg["model_name"]

    def fit(self, texts: list[str]) -> MiniLMEmbedder:
        return self

    def __call__(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.model.encode(texts, batch_size=64, show_progress_bar=False))


class MiniLMOnnxEmbedder:
    name = "minilm_onnx"

    def __init__(self, rcfg: dict):
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        except ImportError as e:
            raise EmbedderUnavailable(f"chromadb not installed: {e}") from e
        fn = ONNXMiniLM_L6_V2()
        target = Path(fn.DOWNLOAD_PATH)
        if not (target / fn.EXTRACTED_FOLDER_NAME / "model.onnx").exists():
            archive = ROOT / rcfg["onnx_archive"]
            if not archive.exists():
                raise EmbedderUnavailable(f"no ONNX model cached and no archive at {archive}")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != fn._MODEL_SHA256:
                raise EmbedderUnavailable(f"{archive} has the wrong SHA-256 ({digest})")
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy(archive, target / fn.ARCHIVE_FILENAME)
            with tarfile.open(target / fn.ARCHIVE_FILENAME, "r:gz") as tar:
                tar.extractall(target, filter="data")
        self.fn = fn
        self.source = f"chromadb ONNX all-MiniLM-L6-v2 ({target})"

    def fit(self, texts: list[str]) -> MiniLMOnnxEmbedder:
        return self

    def __call__(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array(self.fn(texts)))


EMBEDDERS = {"tfidf": TfidfEmbedder, "minilm": MiniLMEmbedder, "minilm_onnx": MiniLMOnnxEmbedder}


def get_embedder(name: str, rcfg: dict):
    if name not in EMBEDDERS:
        raise ValueError(f"unknown embedder {name!r}; choose from {sorted(EMBEDDERS)}")
    return EMBEDDERS[name](rcfg)

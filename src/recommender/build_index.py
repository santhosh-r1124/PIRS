"""Build the ChromaDB index of PAST incidents (training period only).

    python -m src.recommender.build_index                 # embedder from config
    python -m src.recommender.build_index --embedder tfidf

Each training-period ticket is embedded from "module | category | short description" and
stored with its resolution notes. Validation/test-period tickets are never indexed, so the
evaluation in recommend.py retrieves only from the past.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd

from src.data.build_dataset import chronological_split, load_daily, load_source
from src.recommender.embedders import get_embedder
from src.utils import ROOT, load_config


def incident_periods(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(training-period incidents, test-period incidents, boundary dates) of the main split."""
    incidents = load_source(cfg)[0]
    daily = load_daily(cfg)
    ds = cfg["dataset"]
    split = chronological_split(len(daily.dates), ds["lookback"], max(ds["horizons"]), ds["split"])
    train_end, test_start = daily.dates[split.train_end], daily.dates[split.val_end]
    day = incidents["created_at"].dt.normalize()
    train = incidents[day < train_end].reset_index(drop=True)
    test = incidents[day >= test_start].reset_index(drop=True)
    return train, test, {"index_until": str(train_end.date()), "test_from": str(test_start.date())}


def texts(df: pd.DataFrame, template: str) -> list[str]:
    cols = df.fillna("").astype(str)
    return [template.format(**r) for r in cols.to_dict(orient="records")]


def chroma_path(cfg: dict) -> Path:
    """Index folder: $CHROMA_DIR if set (e.g. a local disk when the project lives on a synced
    drive), else recommender.chroma_dir under the project."""
    path = Path(os.environ.get("CHROMA_DIR") or ROOT / cfg["recommender"]["chroma_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def client(cfg: dict):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(chroma_path(cfg)), settings=Settings(anonymized_telemetry=False)
    )


def collection_name(embedder: str) -> str:
    return f"incidents_{embedder}"


def build(cfg: dict, embedder_name: str | None = None, verbose: bool = True):
    """(Re)create the collection for one embedder. Returns (collection, embedder)."""
    rcfg = cfg["recommender"]
    name = embedder_name or rcfg["embedder"]
    train, _, period = incident_periods(cfg)
    docs = texts(train, rcfg["index_text"])
    emb = get_embedder(name, rcfg).fit(docs)
    vectors = emb(docs)

    db = client(cfg)
    cname = collection_name(name)
    if cname in [c if isinstance(c, str) else c.name for c in db.list_collections()]:
        db.delete_collection(cname)
    col = db.create_collection(
        cname, metadata={"hnsw:space": "cosine", **period}, embedding_function=None
    )
    meta = train[["module", "category", "priority", "resolution_notes"]].fillna("").astype(str)
    meta["created_at"] = train["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    rows = meta.to_dict(orient="records")
    for i in range(0, len(train), 2000):  # Chroma limits the batch size
        col.add(
            ids=train["incident_id"].astype(str).tolist()[i : i + 2000],
            embeddings=vectors[i : i + 2000].tolist(),
            documents=train["short_description"].astype(str).tolist()[i : i + 2000],
            metadatas=rows[i : i + 2000],
        )
    if name == "tfidf":  # the fitted vocabulary is needed again at query time (own file)
        with open(chroma_path(cfg) / "tfidf.pkl", "wb") as f:
            pickle.dump(emb, f)
    if verbose:
        print(
            f"indexed {col.count()} training-period incidents (until {period['index_until']}) "
            f"with {name} ({vectors.shape[1]} dims) into {chroma_path(cfg)}/{cname}"
        )
    return col, emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=None)
    args = ap.parse_args()
    build(load_config(), args.embedder)


if __name__ == "__main__":
    main()

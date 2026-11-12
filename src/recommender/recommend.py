"""Resolution recommender: similar past incidents and how they were fixed.

    python -m src.recommender.recommend "Billing documents not posting to FI" --module SD
    python -m src.recommender.recommend --module FI          # fixes for a flagged module
    python -m src.recommender.recommend --evaluate            # hit@5 etc. -> results/

Evaluation (held-out = test-period tickets, index = training-period tickets only): a query is
"module | short description" (the category of a new ticket is unknown); a retrieval counts as
a hit if a retrieved ticket has the same category. Reported against two random baselines
(any past ticket / a past ticket of the same module), computed exactly from category shares.
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd

from src.recommender.build_index import (
    build,
    chroma_path,
    client,
    collection_name,
    incident_periods,
    texts,
)
from src.recommender.embedders import EmbedderUnavailable, get_embedder
from src.utils import get_path, load_config


class Recommender:
    def __init__(self, cfg: dict, embedder_name: str | None = None):
        self.cfg, self.rcfg = cfg, cfg["recommender"]
        self.name = embedder_name or self.rcfg["embedder"]
        try:
            self.col = client(cfg).get_collection(
                collection_name(self.name), embedding_function=None
            )
        except Exception as e:  # chromadb raises its own "does not exist" error
            raise FileNotFoundError(
                f"no recommender index for {self.name!r} in {chroma_path(cfg)}: "
                "run `python run.py index`"
            ) from e
        if self.name == "tfidf":
            with open(chroma_path(cfg) / "tfidf.pkl", "rb") as f:  # own file
                self.embed = pickle.load(f)
        else:
            self.embed = get_embedder(self.name, self.rcfg)

    def _query(self, queries: list[str], k: int, where: dict | None = None) -> list[list[dict]]:
        res = self.col.query(
            query_embeddings=self.embed(queries).tolist(), n_results=k, where=where
        )
        out = []
        for ids, docs, metas, dists in zip(
            res["ids"], res["documents"], res["metadatas"], res["distances"], strict=True
        ):
            out.append(
                [
                    {
                        "incident_id": i,
                        "short_description": d,
                        **m,
                        "similarity": round(1.0 - float(dist), 4),
                    }
                    for i, d, m, dist in zip(ids, docs, metas, dists, strict=True)
                ]
            )
        return out

    def similar(
        self, description: str, module: str = "", k: int | None = None, same_module: bool = False
    ) -> list[dict]:
        """Top-k past incidents for a new ticket description."""
        q = self.rcfg["query_text"].format(module=module, short_description=description)
        where = {"module": module} if same_module and module else None
        return self._query([q], k or self.rcfg["top_k"], where)[0]

    def for_module(
        self, module: str, before: str | None = None, days: int = 90, k: int | None = None
    ) -> dict:
        """Fixes for a module flagged by the forecast: its most frequent recent category
        (in the indexed past) and the most similar past incidents of that kind."""
        got = self.col.get(where={"module": module}, include=["metadatas"])
        meta = pd.DataFrame(got["metadatas"])
        if meta.empty:
            return {"module": module, "typical_category": None, "incidents": []}
        when = pd.to_datetime(meta["created_at"])
        end = (
            min(pd.Timestamp(before), when.max() + pd.Timedelta(days=1))
            if before
            else (when.max() + pd.Timedelta(days=1))
        )
        recent = meta[(when < end) & (when >= end - pd.Timedelta(days=days))]
        top = (recent if len(recent) else meta)["category"].value_counts().index[0]
        hits = self._query(
            [f"{module} | {top.replace('_', ' ')}"],
            k or self.rcfg["top_k"],
            {"$and": [{"module": module}, {"category": top}]},
        )[0]
        return {"module": module, "typical_category": top, "incidents": hits}


# --------------------------------------------------------------------------- evaluation
def random_baselines(train: pd.DataFrame, test: pd.DataFrame, k: int) -> dict:
    """Exact expectations when k past tickets are drawn at random (with replacement)."""
    share = train["category"].value_counts(normalize=True)
    share_m = train.groupby("module")["category"].value_counts(normalize=True)
    p_any = test["category"].map(share).fillna(0).to_numpy()
    p_mod = np.array(
        [share_m.get((m, c), 0.0) for m, c in zip(test["module"], test["category"], strict=True)]
    )
    return {
        "random_any_ticket": {
            "hit@1": float(p_any.mean()),
            f"hit@{k}": float(np.mean(1 - (1 - p_any) ** k)),
            f"precision@{k}": float(p_any.mean()),
        },
        "random_same_module": {
            "hit@1": float(p_mod.mean()),
            f"hit@{k}": float(np.mean(1 - (1 - p_mod) ** k)),
            f"precision@{k}": float(p_mod.mean()),
        },
    }


def evaluate(cfg: dict, embedders: list[str] | None = None) -> dict:
    rcfg, k = cfg["recommender"], cfg["recommender"]["top_k"]
    train, test, period = incident_periods(cfg)
    queries = texts(test, rcfg["query_text"])
    out = {
        "index": {"tickets": len(train), **period},
        "held_out_tickets": len(test),
        "query": rcfg["query_text"],
        "k": k,
        "embedders": {},
        "baselines": random_baselines(train, test, k),
    }
    for name in embedders or rcfg["evaluate_embedders"]:
        try:
            build(cfg, name, verbose=False)
            rec = Recommender(cfg, name)
        except EmbedderUnavailable as e:
            out["embedders"][name] = {"status": f"not available: {e}"}
            continue
        results = []
        for i in range(0, len(queries), 256):
            results += rec._query(queries[i : i + 256], k)
        cats = np.array([[r["category"] for r in rs] for rs in results])
        mods = np.array([[r["module"] for r in rs] for rs in results])
        same = cats == test["category"].to_numpy()[:, None]
        out["embedders"][name] = {
            "status": "ok",
            "source": getattr(rec.embed, "source", name),
            "hit@1": float(same[:, 0].mean()),
            f"hit@{k}": float(same.any(axis=1).mean()),
            f"precision@{k}": float(same.mean()),
            "mrr": float(np.mean([1 / (np.argmax(r) + 1) if r.any() else 0 for r in same])),
            f"same_module_share@{k}": float((mods == test["module"].to_numpy()[:, None]).mean()),
            "mean_top1_similarity": float(np.mean([rs[0]["similarity"] for rs in results])),
            f"hit@{k}_by_category": {
                c: float(same[(test["category"] == c).to_numpy()].any(axis=1).mean())
                for c in sorted(test["category"].unique())
            },
        }
    return out


def write_report(cfg: dict, ev: dict) -> None:
    rd = get_path(cfg, "results")
    (rd / "recommender.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    k = ev["k"]
    rows = [
        f"| {n} | "
        + (
            " | ".join(f"{m[x]:.3f}" for x in ["hit@1", f"hit@{k}", f"precision@{k}", "mrr"])
            if m["status"] == "ok"
            else m["status"] + " | | |"
        )
        + " |"
        for n, m in ev["embedders"].items()
    ]
    rows += [
        f"| {n} (expected) | {m['hit@1']:.3f} | {m[f'hit@{k}']:.3f} | "
        f"{m[f'precision@{k}']:.3f} | - |"
        for n, m in ev["baselines"].items()
    ]
    md = [
        "# Resolution recommender (SYNTHETIC DATA)",
        "",
        f"Index: {ev['index']['tickets']} training-period tickets (until "
        f"{ev['index']['index_until']}). Held out: {ev['held_out_tickets']} test-period "
        f"tickets (from {ev['index']['test_from']}). Query: `{ev['query']}`. A hit = a "
        f"retrieved past ticket has the same category (8 categories).",
        "",
        f"| retriever | hit@1 | hit@{k} | precision@{k} | MRR |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]
    (rd / "recommender.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("description", nargs="?")
    ap.add_argument("--module", default="")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.evaluate:
        write_report(cfg, evaluate(cfg))
        return
    rec = Recommender(cfg, args.embedder)
    hits = (
        rec.similar(args.description, args.module)
        if args.description
        else rec.for_module(args.module)["incidents"]
    )
    for h in hits:
        print(
            f"{h['similarity']:.3f}  [{h['module']}/{h['category']}] {h['short_description']}"
            f"\n       fix: {h['resolution_notes']}"
        )


if __name__ == "__main__":
    main()

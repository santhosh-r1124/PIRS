"""Streamlit dashboard.

    python run.py api          # first, in another terminal
    python run.py dashboard    # then open http://localhost:8501

Talks to the API at $API_URL (default http://localhost:8000). The model-comparison page
reads the tables and figures that `python run.py evaluate` wrote to results/.
"""

from __future__ import annotations

import datetime as dt
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src.graph.sap_graph import LAYOUT, build_graph  # noqa: E402
from src.utils import ROOT, load_config  # noqa: E402

API = os.environ.get("API_URL", "http://localhost:8000")
CFG = load_config()
RESULTS = ROOT / CFG["paths"]["results"]


def api(method: str, path: str, **kwargs) -> dict:
    r = requests.request(method, f"{API}{path}", timeout=120, **kwargs)
    if r.status_code >= 400:
        detail = (
            r.json().get("detail", r.text)
            if r.headers.get("content-type", "").startswith("application/json")
            else r.text
        )
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def risk_graph(forecast: dict):
    graph = build_graph(CFG)
    g = graph.to_networkx()
    risk = {m["module"]: m["p1p2_risk"] for m in forecast["modules"]}
    fig, ax = plt.subplots(figsize=(7, 5))
    nodes = nx.draw_networkx_nodes(
        g,
        LAYOUT,
        node_size=1400,
        cmap="Reds",
        vmin=0,
        vmax=1,
        node_color=[risk[n] for n in g.nodes],
        edgecolors="#555",
        ax=ax,
    )
    nx.draw_networkx_labels(g, LAYOUT, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        g,
        LAYOUT,
        node_size=1400,
        arrowsize=10,
        edge_color="#aaa",
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    for m in forecast["top3"]:
        x, y = LAYOUT[m]
        ax.add_patch(plt.Circle((x, y), 0.45, fill=False, lw=3, ec="#1f4e79"))
    fig.colorbar(nodes, ax=ax, label="P1/P2 risk score", fraction=0.04)
    ax.set_title(
        f"Risk for {forecast['date']} (forecast made {forecast['origin_date']}); blue ring = top 3",
        fontsize=9,
    )
    ax.axis("off")
    return fig


def forecast_page(health: dict) -> None:
    first, last = (dt.date.fromisoformat(d) for d in health["forecast_dates"])
    default = min(max(dt.date(2025, 11, 28), first), last)
    c1, c2 = st.sidebar.columns(2)
    day = c1.date_input("Target day", value=default, min_value=first, max_value=last)
    horizon = c2.selectbox("Days ahead", list(range(1, 8)), index=0)
    try:
        fc = api("GET", "/forecast", params={"date": str(day), "horizon": horizon})
    except RuntimeError as e:
        st.error(str(e))
        return
    st.subheader(f"Forecast for {fc['date']} ({horizon} day(s) ahead, model: {fc['model']})")
    left, right = st.columns([1.2, 1])
    left.pyplot(risk_graph(fc))
    table = pd.DataFrame(fc["modules"]).sort_values("p1p2_risk", ascending=False)
    right.dataframe(table, hide_index=True, use_container_width=True)
    right.caption(fc["risk_note"])

    st.subheader("Top 3 modules: why, and how similar incidents were fixed")
    for m in fc["top3"]:
        with st.expander(
            f"{m} - risk {next(x['p1p2_risk'] for x in fc['modules'] if x['module'] == m):.2f}",
            expanded=m == fc["top3"][0],
        ):
            a, b = st.columns(2)
            ex = api(
                "GET", "/explain", params={"module": m, "date": fc["date"], "horizon": horizon}
            )
            infl = pd.DataFrame(ex.get("influences", []))
            a.markdown("**Most influential neighbouring modules** (graph attention)")
            if len(infl):
                a.bar_chart(infl.set_index("source")["weight"])
            else:
                a.write("No incoming edges.")
            a.caption(ex["note"])
            rec = api("GET", "/recommend/module", params={"module": m, "date": fc["origin_date"]})
            b.markdown(f"**Typical recent problem:** `{rec['typical_category']}`")
            for h in rec["incidents"][:3]:
                b.markdown(f"- *{h['short_description']}*  \n  fix: {h['resolution_notes']}")

    st.subheader("Forecast vs actual (test period, never used for training)")
    mod = st.selectbox("Module", health["modules"], index=0)
    hist = api("GET", "/history", params={"module": mod, "horizon": horizon})
    st.line_chart(
        pd.DataFrame(
            {"forecast (expected)": hist["expected"], "actual": hist["actual"]},
            index=pd.to_datetime(hist["dates"]),
        )
    )

    st.subheader("New ticket: similar past incidents and their fixes")
    q1, q2 = st.columns([3, 1])
    text = q1.text_input("Ticket description", "Invoices from billing not reaching accounting")
    qmod = q2.selectbox("Module", [""] + health["modules"], index=3)
    if text.strip():
        res = api("POST", "/recommend", json={"description": text, "module": qmod, "k": 5})
        st.dataframe(
            pd.DataFrame(res["results"])[
                ["similarity", "module", "category", "short_description", "resolution_notes"]
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(f"Retriever: {res['embedder']} over training-period tickets.")


def comparison_page() -> None:
    st.subheader("Model comparison (test split, mean ± std over seeds)")
    for target, cols in [
        ("regression", ["MAE@1", "MAE@7", "PoissonDev@1", "P@3@1"]),
        ("classification", ["PR_AUC@1", "ROC_AUC@1", "P@3@1", "R@3@1"]),
    ]:
        path = RESULTS / f"comparison_{target}.csv"
        if not path.exists():
            st.info(f"{path.name} not found: run `python run.py train`.")
            continue
        df = pd.read_csv(path)
        show = df[["model", "n_seeds"]].copy()
        for c in cols:
            show[c] = [
                f"{m:.3f}" + ("" if pd.isna(s) else f" ± {s:.3f}")
                for m, s in zip(df[f"{c}_mean"], df[f"{c}_std"], strict=True)
            ]
        st.markdown(
            f"**{'Incident counts' if target == 'regression' else 'P1/P2 incidents'}**"
            " (MAE / deviance: lower is better)"
        )
        st.dataframe(show, hide_index=True, use_container_width=True)
    for name, caption in [
        ("horizon_mae.png", "Error by forecast horizon"),
        ("rolling_origin.png", "Stability over 4 test windows"),
        ("explain_adjacency.png", "What the models learned vs the truth"),
    ]:
        p = RESULTS / "figures" / name
        if p.exists():
            st.image(str(p), caption=caption)
    st.caption("Full analysis: results/summary.md (oracle ceiling, ablations, explainability).")


def main() -> None:
    st.set_page_config(page_title="SAP incident forecasting", layout="wide")
    st.title("SAP incident forecasting")
    try:
        health = api("GET", "/health")
    except Exception as e:  # API down: say so instead of crashing
        st.error(f"The API at {API} is not reachable ({e}). Start it with `python run.py api`.")
        st.stop()
        return  # st.stop() halts Streamlit; return also keeps plain imports safe
    if health.get("synthetic", True):
        st.warning(
            "SYNTHETIC DATA - every ticket, forecast and metric here comes from "
            "generated data, not from a real SAP system."
        )
    if health["status"] != "ok":
        st.error(health.get("detail", "API not ready"))
        st.stop()
        return  # st.stop() halts Streamlit; return also keeps plain imports safe
    page = st.sidebar.radio("Page", ["Forecast", "Model comparison"])
    forecast_page(health) if page == "Forecast" else comparison_page()


if __name__ == "__main__":  # Streamlit runs the script as __main__
    main()

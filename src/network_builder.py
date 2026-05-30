"""
Phase 2 — Network construction.

Two graph families, both driven from the Parquet dumps (no XML re-read):

  1. Tag co-occurrence graph (undirected, weighted)
        node  = tag
        edge  = (t1, t2) if they appear together in at least one question
        w     = # questions containing both t1 and t2

  2. User Q&A graph (directed, weighted)
        node  = user_id (Stack Overflow OwnerUserId)
        edge  = answerer_id -> questioner_id  (knowledge flows FROM answerer
                TO questioner — we chose this direction deliberately; see
                report/phase2_record.md for the discussion)
        w     = # times answerer answered this questioner's questions

Both can be scoped to a community (`in_<c>` = True) or to a date window.

Polars drives the aggregations; NetworkX is only used at the very end to
wrap the edge list for downstream SNA / DNA code.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Iterable

import networkx as nx
import polars as pl

from data_loader import (
    Q_PATH,
    A_PATH,
    COMMUNITIES,
    load_answers,
    load_questions,
)

ROOT = Path(__file__).resolve().parent.parent
NET_DIR = ROOT / "data" / "networks"
NET_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tag co-occurrence
# ---------------------------------------------------------------------------

def tag_cooccurrence_edges(
    community: str | None = None,
    llm_only: bool = False,
    from_date: str | None = None,
    to_date: str | None = None,
    min_weight: int = 5,
    restrict_to_pool: set[str] | None = None,
) -> pl.DataFrame:
    """
    Build tag co-occurrence edge list with polars.

    Strategy (fast, vectorised):
        - scan_parquet with pushed-down filters
        - keep only (Id, Tags) columns
        - explode Tags → (Id, tag) rows
        - self-join on Id, keep only (t1, t2) with t1 < t2 → unordered pairs
        - group_by (t1, t2), count → weight

    Args:
        community, llm_only, from_date, to_date: filters matching
            data_loader.load_questions semantics.
        min_weight: drop pairs whose co-occurrence count is below this.
            Default 5 follows Ye, Xing & Kapre (2017); for small community
            subsets use 2–3.
        restrict_to_pool: if given, only edges where BOTH endpoints are in
            this set are kept. Useful for building a pure ai_ml tag graph
            without leakage from bridge tags (python, java, ...).

    Returns:
        Polars DataFrame with columns [t1, t2, weight].
    """
    lf = pl.scan_parquet(Q_PATH).select(["Id", "Tags", "CreationDate",
                                         *[f"in_{c}" for c in COMMUNITIES],
                                         "in_llm"])
    if community is not None:
        if community not in COMMUNITIES:
            raise ValueError(f"community must be one of {COMMUNITIES}")
        lf = lf.filter(pl.col(f"in_{community}"))
    if llm_only:
        lf = lf.filter(pl.col("in_llm"))
    if from_date is not None:
        lf = lf.filter(pl.col("CreationDate") >= from_date)
    if to_date is not None:
        lf = lf.filter(pl.col("CreationDate") < to_date)

    lf = lf.select(["Id", "Tags"])

    # explode tags → (Id, tag)
    exploded = lf.explode("Tags").rename({"Tags": "tag"}).filter(
        pl.col("tag").is_not_null()
    )

    if restrict_to_pool is not None:
        exploded = exploded.filter(pl.col("tag").is_in(list(restrict_to_pool)))

    # self-join on Id to get all pairs; keep t1 < t2 for undirected
    left = exploded.rename({"tag": "t1"})
    right = exploded.rename({"tag": "t2"})
    pairs = left.join(right, on="Id", how="inner").filter(
        pl.col("t1") < pl.col("t2")
    )

    edges = (
        pairs.group_by(["t1", "t2"])
        .agg(pl.len().alias("weight"))
        .filter(pl.col("weight") >= min_weight)
        .sort("weight", descending=True)
        .collect()
    )
    return edges


def tag_cooccurrence_graph(
    community: str | None = None,
    llm_only: bool = False,
    from_date: str | None = None,
    to_date: str | None = None,
    min_weight: int = 5,
    restrict_to_pool: set[str] | None = None,
) -> nx.Graph:
    """Return an undirected NetworkX Graph with edge attribute 'weight'."""
    edges = tag_cooccurrence_edges(
        community=community, llm_only=llm_only,
        from_date=from_date, to_date=to_date,
        min_weight=min_weight, restrict_to_pool=restrict_to_pool,
    )
    G = nx.Graph()
    for row in edges.iter_rows(named=True):
        G.add_edge(row["t1"], row["t2"], weight=row["weight"])
    return G


# ---------------------------------------------------------------------------
# User Q&A network
# ---------------------------------------------------------------------------

def user_qa_edges(
    community: str | None = None,
    llm_only: bool = False,
    from_date: str | None = None,
    to_date: str | None = None,
    min_weight: int = 1,
) -> pl.DataFrame:
    """
    Build directed user Q&A edge list (answerer -> questioner).

    Only keeps answers/questions where BOTH owners are non-null and distinct
    (a user answering their own question is dropped: it's not knowledge flow).

    Scoping:
        - community/llm_only filter is applied to QUESTIONS; only answers
          whose ParentId lands in the filtered question set are kept.
        - from_date/to_date filters the ANSWER creation date. This matches
          the intuition that "the edge happened when the answer was posted."

    Returns Polars DataFrame with columns
        [answerer_id, questioner_id, weight].
    """
    # 1. scope the questions
    q_lf = pl.scan_parquet(Q_PATH).select(["Id", "OwnerUserId",
                                           *[f"in_{c}" for c in COMMUNITIES],
                                           "in_llm"])
    if community is not None:
        q_lf = q_lf.filter(pl.col(f"in_{community}"))
    if llm_only:
        q_lf = q_lf.filter(pl.col("in_llm"))

    q_lf = (q_lf.filter(pl.col("OwnerUserId").is_not_null())
            .select([pl.col("Id").alias("ParentId"),
                     pl.col("OwnerUserId").alias("questioner_id")]))

    # 2. load relevant answers
    a_lf = pl.scan_parquet(A_PATH).select(["ParentId", "OwnerUserId",
                                           "CreationDate"])
    if from_date is not None:
        a_lf = a_lf.filter(pl.col("CreationDate") >= from_date)
    if to_date is not None:
        a_lf = a_lf.filter(pl.col("CreationDate") < to_date)
    a_lf = (a_lf.filter(pl.col("OwnerUserId").is_not_null())
            .select([pl.col("ParentId"),
                     pl.col("OwnerUserId").alias("answerer_id")]))

    # 3. join, drop self-answers, aggregate
    edges = (
        a_lf.join(q_lf, on="ParentId", how="inner")
        .filter(pl.col("answerer_id") != pl.col("questioner_id"))
        .group_by(["answerer_id", "questioner_id"])
        .agg(pl.len().alias("weight"))
        .filter(pl.col("weight") >= min_weight)
        .collect()
    )
    return edges


def user_qa_graph(
    community: str | None = None,
    llm_only: bool = False,
    from_date: str | None = None,
    to_date: str | None = None,
    min_weight: int = 1,
) -> nx.DiGraph:
    """Return a directed NetworkX graph with 'weight' edge attribute."""
    edges = user_qa_edges(community=community, llm_only=llm_only,
                          from_date=from_date, to_date=to_date,
                          min_weight=min_weight)
    G = nx.DiGraph()
    # Add edges in one shot — way faster than row-by-row for millions of edges
    edge_iter = (
        (r["answerer_id"], r["questioner_id"], {"weight": r["weight"]})
        for r in edges.iter_rows(named=True)
    )
    G.add_edges_from(edge_iter)
    return G


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_graph(G: nx.Graph, name: str) -> Path:
    """Pickle the graph. Pickle is ~10x faster than GraphML and round-trips
    all attribute types, which NetworkX's GraphML writer chokes on."""
    path = NET_DIR / f"{name}.pkl"
    with path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_graph(name: str) -> nx.Graph | nx.DiGraph:
    path = NET_DIR / f"{name}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def summarize_graph(G: nx.Graph | nx.DiGraph) -> dict:
    """Cheap stats that fit in a one-row summary."""
    directed = G.is_directed()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {"n_nodes": 0, "n_edges": 0}

    degrees = dict(G.degree())
    weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    total_w = sum(weights)
    max_deg = max(degrees.values())
    return {
        "directed": directed,
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(G),
        "total_weight": total_w,
        "mean_weight": total_w / m if m else 0,
        "max_weight": max(weights) if weights else 0,
        "max_degree": max_deg,
        "mean_degree": sum(degrees.values()) / n,
    }


if __name__ == "__main__":
    # Smoke test: ai_ml pilot
    t0 = time.perf_counter()
    print("Running smoke test on ai_ml community ...")
    G_tag = tag_cooccurrence_graph(community="ai_ml", min_weight=5)
    print(f"  tag graph: {summarize_graph(G_tag)} [{time.perf_counter()-t0:.1f}s]")

    t1 = time.perf_counter()
    G_user = user_qa_graph(community="ai_ml", min_weight=1)
    print(f"  user graph: {summarize_graph(G_user)} [{time.perf_counter()-t1:.1f}s]")

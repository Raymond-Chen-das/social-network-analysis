"""
Empirical timing test for null-model rewiring strategies.

Measures wall time for:
  1. nx.double_edge_swap (Maslov-Sneppen, current approach)
  2. nx.configuration_model + cleanup (proposed C-backed alternative)

Run on the smallest tag graph (ai_ml) first, then extrapolate.

This script's only purpose is to GIVE EVIDENCE for runtime estimates,
not to produce final results. Do not delete after use — keep as a
reproducibility artefact.
"""

from __future__ import annotations

import time
from pathlib import Path

import community as community_louvain
import networkx as nx

from network_builder import load_graph

ROOT = Path(__file__).resolve().parent.parent


def time_double_edge_swap(G: nx.Graph, swap_ratio: int = 2,
                           seed: int = 42) -> tuple[float, float]:
    """Returns (rewire_time, louvain_time) in seconds."""
    G_copy = G.copy()
    n_swaps = G.number_of_edges() * swap_ratio

    t0 = time.perf_counter()
    nx.double_edge_swap(G_copy, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
    t_rewire = time.perf_counter() - t0

    t0 = time.perf_counter()
    part = community_louvain.best_partition(G_copy, weight="weight",
                                             random_state=seed)
    _ = community_louvain.modularity(part, G_copy, weight="weight")
    t_louvain = time.perf_counter() - t0

    return t_rewire, t_louvain


def time_configuration_model(G: nx.Graph, seed: int = 42) -> tuple[float, float]:
    """Configuration model + cleanup. Returns (rewire_time, louvain_time)."""
    deg_seq = [d for _, d in G.degree()]

    t0 = time.perf_counter()
    G_null = nx.configuration_model(deg_seq, seed=seed)
    G_null = nx.Graph(G_null)  # collapse multi-edges
    G_null.remove_edges_from(nx.selfloop_edges(G_null))
    # No weights — set all to 1 so Louvain runs in unweighted mode
    for u, v in G_null.edges():
        G_null[u][v]["weight"] = 1
    t_rewire = time.perf_counter() - t0

    t0 = time.perf_counter()
    part = community_louvain.best_partition(G_null, weight="weight",
                                             random_state=seed)
    _ = community_louvain.modularity(part, G_null, weight="weight")
    t_louvain = time.perf_counter() - t0

    return t_rewire, t_louvain


def main() -> None:
    print("Loading graphs ...", flush=True)
    targets = ["databases", "mobile", "web_frontend"]
    graphs = {c: load_graph(f"tag_cooc_{c}") for c in targets}
    for c, g in graphs.items():
        print(f"  {c}: {g.number_of_nodes():,} nodes, "
              f"{g.number_of_edges():,} edges", flush=True)

    print("\n=== Method A: double_edge_swap (current) ===", flush=True)
    print(f"{'Graph':<15} {'rewire':>10} {'louvain':>10} {'total':>10}",
          flush=True)
    for c, g in graphs.items():
        t_rw, t_lv = time_double_edge_swap(g, swap_ratio=2)
        print(f"{c:<15} {t_rw:>9.1f}s {t_lv:>9.1f}s {t_rw+t_lv:>9.1f}s",
              flush=True)

    print("\n=== Method B: configuration_model (proposed) ===", flush=True)
    print(f"{'Graph':<15} {'rewire':>10} {'louvain':>10} {'total':>10}",
          flush=True)
    for c, g in graphs.items():
        t_rw, t_lv = time_configuration_model(g)
        print(f"{c:<15} {t_rw:>9.1f}s {t_lv:>9.1f}s {t_rw+t_lv:>9.1f}s",
              flush=True)

    print("\nExtrapolation (30 rewirings per graph, 6 graphs total):", flush=True)
    print("  Multiply Method-X total by 30 then sum across graphs.")


if __name__ == "__main__":
    main()

"""
Recompute bridge_users for ALL 5 community user_qa graphs using EXACT
betweenness centrality on the auto-tuned k-core.

Why exact instead of sampled?
  Sampled betweenness (`networkx.betweenness_centrality(k=500, seed=42)`)
  selects pivot nodes BY INDEX from the graph's node iterator. Across
  re-runs, the node insertion order can differ (depending on how the
  underlying directed graph was loaded and converted to undirected),
  giving different pivot sets and therefore different betweenness
  estimates for the same nodes. This breaks reproducibility.

  Exact betweenness has no sampling, runs over all source nodes, and
  is fully deterministic. At core sizes ≤ 3,000 nodes (which is what
  our k-core auto-tuning targets), exact betweenness completes in
  10-60 seconds per graph.

Outputs:
  outputs/tables/bridge_users_<community>.csv (overwritten)
  outputs/tables/bridge_users_qa_log.txt       (verification record)
"""

from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import pandas as pd

from network_builder import load_graph

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "tables"

COMMUNITIES = ("ai_ml", "web_frontend", "mobile", "cloud_devops", "databases")


def compute_bridge_users_exact(G: nx.DiGraph,
                                top_n: int = 50,
                                max_core_nodes: int = 3_000
                                ) -> tuple[pd.DataFrame, int, int]:
    """
    K-core auto-tune to <= max_core_nodes, then EXACT betweenness.

    Returns (top_n_dataframe, k_used, core_node_count).
    """
    G_und = G.to_undirected(reciprocal=False)

    k = 5
    core = nx.k_core(G_und, k=k)
    while len(core) > max_core_nodes:
        k += 5
        core = nx.k_core(G_und, k=k)
        if k > 200:
            break
    if len(core) < 20:
        for k_fall in range(k - 1, 0, -1):
            core = nx.k_core(G_und, k=k_fall)
            if len(core) >= 20:
                break

    # EXACT betweenness — no k= argument, no seed
    bet = nx.betweenness_centrality(core, normalized=True)

    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    rows = [
        {
            "user_id": node,
            "betweenness": bet[node],
            "out_degree":  out_deg.get(node, 0),
            "in_degree":   in_deg.get(node, 0),
            "total_degree": out_deg.get(node, 0) + in_deg.get(node, 0),
        }
        for node in core.nodes()
    ]
    df = pd.DataFrame(rows).sort_values("betweenness", ascending=False)
    return df.head(top_n).reset_index(drop=True), k, len(core)


def main() -> None:
    log_lines = ["# bridge_users repair log",
                 "Method: EXACT betweenness on k-core auto-tuned to <= 3000 nodes.",
                 ""]
    for community in COMMUNITIES:
        print(f"\n=== {community} ===", flush=True)
        t0 = time.perf_counter()
        G = load_graph(f"user_qa_{community}")
        print(f"  graph: {G.number_of_nodes():,} nodes  "
              f"{G.number_of_edges():,} edges", flush=True)

        df, k_used, core_size = compute_bridge_users_exact(G, top_n=50)
        elapsed = time.perf_counter() - t0

        path = OUT_DIR / f"bridge_users_{community}.csv"
        df.to_csv(path, index=False)

        top_user = int(df["user_id"].iloc[0])
        top_bet = df["betweenness"].iloc[0]

        msg = (f"  k={k_used}  core={core_size:,} nodes  "
               f"top_user={top_user}  top_bet={top_bet:.4f}  "
               f"elapsed={elapsed/60:.1f} min")
        print(msg, flush=True)
        log_lines.append(f"{community}: {msg.strip()}")

        del G

    (OUT_DIR / "bridge_users_qa_log.txt").write_text("\n".join(log_lines),
                                                      encoding="utf-8")
    print("\nDone.")


if __name__ == "__main__":
    main()

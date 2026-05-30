"""
Phase 3 — Social Network Analysis.

Tasks
-----
1. Louvain community detection on each tag co-occurrence graph.
2. NMI between Louvain partition and our manual taxonomy (ground truth).
3. Bridge-user identification on user Q&A directed graphs.
4. Basic centrality stats per community.

Design notes
------------
- Louvain is run on tag GRAPHS (not user graphs) because:
    (a) tag graphs are small enough (≤ 44k nodes, ≤ 630k edges) for
        networkx community detection to finish in seconds;
    (b) the NMI comparison requires nodes to have known taxonomy labels,
        which only tags have (via target_tags_proposed.json).

- For user graphs, we compute approximate betweenness on a k-core subgraph
  to keep runtime tractable (betweenness is O(VE) on the full graph).

- random_state=42 everywhere for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from network_builder import load_graph, NET_DIR

ROOT = Path(__file__).resolve().parent.parent
TAGS_JSON = ROOT / "outputs" / "tables" / "target_tags_proposed.json"
OUT_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COMMUNITIES = ("ai_ml", "web_frontend", "mobile", "cloud_devops", "databases")

# ---------------------------------------------------------------------------
# Load taxonomy ground truth
# ---------------------------------------------------------------------------

def load_taxonomy() -> dict[str, str]:
    """Return {tag: community_name} for all tags in target_tags_proposed.json."""
    raw = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
    tag_to_community: dict[str, str] = {}
    for community, tags in raw["target_tags"].items():
        for t in tags:
            tag_to_community[t] = community
    return tag_to_community


# ---------------------------------------------------------------------------
# Louvain + NMI
# ---------------------------------------------------------------------------

def run_louvain(G: nx.Graph, resolution: float = 1.0,
                random_state: int = 42) -> tuple[dict, float]:
    """
    Run Louvain on a weighted undirected graph.

    Returns
    -------
    partition : {node: community_id}
    modularity : float
    """
    partition = community_louvain.best_partition(
        G, weight="weight", resolution=resolution, random_state=random_state
    )
    modularity = community_louvain.modularity(partition, G, weight="weight")
    return partition, modularity


def compute_nmi(partition: dict[str, int],
                taxonomy: dict[str, str],
                nodes: list[str] | None = None) -> float:
    """
    NMI between Louvain partition and manual taxonomy.

    Only nodes present in BOTH partition and taxonomy are used.
    Nodes not in taxonomy (general-purpose tags like 'python') are skipped.

    Returns NMI in [0, 1]. Higher = better alignment.
    """
    if nodes is None:
        nodes = list(partition.keys())

    overlap = [n for n in nodes if n in taxonomy and n in partition]
    if len(overlap) < 10:
        return float("nan")

    louvain_labels = [partition[n] for n in overlap]
    taxonomy_labels = [taxonomy[n] for n in overlap]

    return normalized_mutual_info_score(taxonomy_labels, louvain_labels,
                                        average_method="arithmetic")


def louvain_community_summary(partition: dict[str, int],
                               taxonomy: dict[str, str]) -> pd.DataFrame:
    """
    For each Louvain community, report the dominant taxonomy label and
    coverage stats.
    """
    rows = []
    community_ids = sorted(set(partition.values()))
    for cid in community_ids:
        members = [n for n, c in partition.items() if c == cid]
        labeled = [taxonomy[n] for n in members if n in taxonomy]
        if not labeled:
            dominant = "unknown"
            purity = 0.0
        else:
            from collections import Counter
            cnt = Counter(labeled)
            dominant, top_count = cnt.most_common(1)[0]
            purity = top_count / len(labeled)
        rows.append({
            "louvain_id": cid,
            "n_members": len(members),
            "n_labeled": len(labeled),
            "dominant_taxonomy": dominant,
            "purity": purity,
        })
    return pd.DataFrame(rows).sort_values("n_members", ascending=False)


# ---------------------------------------------------------------------------
# Bridge-user analysis on user Q&A graphs
# ---------------------------------------------------------------------------

def compute_bridge_users(G: nx.DiGraph,
                         top_n: int = 50,
                         max_core_nodes: int = 3_000,
                         n_samples: int = 500,
                         large_graph_threshold: int = 500_000) -> pd.DataFrame:
    """
    Identify bridge users in a directed user Q&A graph.

    For graphs with <= large_graph_threshold nodes:
        Auto-tune k-core to <= max_core_nodes, then run approximate
        betweenness centrality (k=n_samples random pivots).

    For graphs above large_graph_threshold nodes (e.g. global):
        Skip betweenness entirely. Use top out-degree as proxy —
        high out-degree = answered many distinct users = strong knowledge
        broadcaster. No undirected copy needed; runs in O(V).

    Returns top_n nodes sorted by betweenness (or out_degree for large graphs).
    """
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    if G.number_of_nodes() > large_graph_threshold:
        print(f"    large graph ({G.number_of_nodes():,} nodes): "
              f"using out-degree proxy", flush=True)
        rows = [
            {
                "user_id": node,
                "betweenness": float("nan"),  # not computed
                "out_degree":  out_deg.get(node, 0),
                "in_degree":   in_deg.get(node, 0),
                "total_degree": out_deg.get(node, 0) + in_deg.get(node, 0),
            }
            for node in G.nodes()
            if out_deg.get(node, 0) > 0
        ]
        df = pd.DataFrame(rows).sort_values("out_degree", ascending=False)
        return df.head(top_n).reset_index(drop=True)

    # --- small / medium graphs: betweenness on k-core ---
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

    print(f"    k-core k={k}: {len(core):,} nodes", flush=True)

    bet = nx.betweenness_centrality(core, k=min(n_samples, len(core)),
                                    normalized=True, seed=42)
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
    return df.head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full SNA run for one community
# ---------------------------------------------------------------------------

def analyze_community(community: str,
                      taxonomy: dict[str, str],
                      resolutions: tuple[float, ...] = (0.5, 1.0, 1.5)) -> dict:
    """
    Full SNA analysis for a single community (or 'global').

    Returns a dict with all metrics. Saves per-community CSVs.
    """
    tag_name  = f"tag_cooc_{community}"
    user_name = f"user_qa_{community}"

    print(f"\n[{community}] loading graphs ...", flush=True)
    G_tag  = load_graph(tag_name)
    G_user = load_graph(user_name)

    print(f"  tag:  {G_tag.number_of_nodes():,} nodes  "
          f"{G_tag.number_of_edges():,} edges", flush=True)
    print(f"  user: {G_user.number_of_nodes():,} nodes  "
          f"{G_user.number_of_edges():,} edges", flush=True)

    # --- Louvain sweep over resolutions ------------------------------------
    louvain_results = {}
    for res in resolutions:
        partition, modularity = run_louvain(G_tag, resolution=res)
        nmi = compute_nmi(partition, taxonomy)
        n_communities_detected = len(set(partition.values()))
        louvain_results[res] = {
            "resolution": res,
            "modularity": modularity,
            "n_communities": n_communities_detected,
            "nmi_vs_taxonomy": nmi,
            "partition": partition,
        }
        print(f"  Louvain res={res:.1f}: Q={modularity:.4f} "
              f"n_comm={n_communities_detected} NMI={nmi:.4f}", flush=True)

    # Best resolution by modularity
    best_res = max(louvain_results, key=lambda r: louvain_results[r]["modularity"])
    best = louvain_results[best_res]

    # Community summary table for best partition
    summary_df = louvain_community_summary(best["partition"], taxonomy)
    summary_path = OUT_DIR / f"louvain_{community}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Community summary → {summary_path.name}", flush=True)

    # --- Bridge users -------------------------------------------------------
    print(f"  Computing bridge users ...", flush=True)
    bridge_df = compute_bridge_users(G_user, top_n=50, n_samples=500)
    bridge_path = OUT_DIR / f"bridge_users_{community}.csv"
    bridge_df.to_csv(bridge_path, index=False)
    print(f"  Bridge users → {bridge_path.name} "
          f"(top bet={bridge_df['betweenness'].iloc[0]:.4f})", flush=True)

    return {
        "community": community,
        "tag_nodes": G_tag.number_of_nodes(),
        "tag_edges": G_tag.number_of_edges(),
        "user_nodes": G_user.number_of_nodes(),
        "user_edges": G_user.number_of_edges(),
        "best_resolution": best_res,
        "modularity": best["modularity"],
        "n_louvain_communities": best["n_communities"],
        "nmi_vs_taxonomy": best["nmi_vs_taxonomy"],
        "top_bridge_user_id": int(bridge_df["user_id"].iloc[0]) if len(bridge_df) else None,
        "top_bridge_betweenness": float(bridge_df["betweenness"].iloc[0]) if len(bridge_df) else None,
    }


# ---------------------------------------------------------------------------
# Main — run all 5 communities + global
# ---------------------------------------------------------------------------

def main() -> None:
    taxonomy = load_taxonomy()
    print(f"Taxonomy loaded: {len(taxonomy)} tagged entries "
          f"across {len(set(taxonomy.values()))} communities", flush=True)

    results = []
    targets = list(COMMUNITIES) + ["global"]
    for c in targets:
        try:
            r = analyze_community(c, taxonomy)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR on {c}: {exc}", flush=True)

    # Summary table
    df = pd.DataFrame(results)
    summary_path = OUT_DIR / "sna_summary.csv"
    df.to_csv(summary_path, index=False)

    # Markdown table
    lines = ["# SNA Summary\n",
             "| Community | Tag N | Tag E | Louvain Q | NMI | "
             "Louvain #comm | User N | User E |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in df.iterrows():
        lines.append(
            f"| `{row['community']}` | {row['tag_nodes']:,} | "
            f"{row['tag_edges']:,} | {row['modularity']:.4f} | "
            f"{row['nmi_vs_taxonomy']:.4f} | {row['n_louvain_communities']} | "
            f"{row['user_nodes']:,} | {row['user_edges']:,} |"
        )
    md_path = OUT_DIR / "sna_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSNA summary → {summary_path}")
    print(f"SNA summary (md) → {md_path}")
    cols = [c for c in ["community", "modularity", "nmi_vs_taxonomy",
                         "n_louvain_communities"] if c in df.columns]
    if cols:
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

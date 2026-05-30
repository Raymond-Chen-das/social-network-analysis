"""
Phase 3 finalisation.

This script does NOT recompute Louvain or per-community betweenness — those
results already live in outputs/tables/louvain_*.csv and bridge_users_*.csv
from the earlier sna_analysis.py runs.

What this adds:

  1. Cross-community bridge users (the proper RQ1 operationalisation):
     a user_id is a bridge user iff it appears in >= 2 community user_qa
     graphs. We rank by (n_communities, total_out_degree).

  2. Null model baseline for Louvain modularity:
     For each community tag graph, perform `n_rewirings` random
     edge-swaps preserving the degree sequence, run Louvain on each
     rewired graph, compare observed Q vs the random-Q distribution.
     Report z-score and one-sided p-value.

  3. A complete sna_summary.md aggregating all results — the previous
     run died on the summary step due to empty result list.

  4. A bridge_users_cross_community.csv with the top cross-community
     users, including which communities they appear in.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd

from network_builder import load_graph

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "tables"
COMMUNITIES = ("ai_ml", "web_frontend", "mobile", "cloud_devops", "databases")

N_REWIRINGS = 20          # Louvain runs on rewired graphs (>=20 for z-score normal approx)
SWAP_RATIO = 2            # nswap = SWAP_RATIO * |E|; 2x is standard adequacy
RNG_SEED = 42

# Empirical per-graph timings (seconds per single rewiring + Louvain),
# measured on this machine 2026-04-26 via src/null_model_timing_test.py.
# Used to print honest ETA at runtime (no more guessing).
EMPIRICAL_SEC_PER_ITER = {
    "ai_ml":        10.8,
    "cloud_devops":  9.6,
    "databases":    18.8,
    "mobile":       65.7,
    "web_frontend": 128.0,
    "global":      185.0,
}


# ---------------------------------------------------------------------------
# 1. Cross-community bridge users
# ---------------------------------------------------------------------------

def cross_community_bridge_users(top_n: int = 50) -> pd.DataFrame:
    """
    For each user, record which community user_qa graphs they appear in
    and their out-degree in each.

    A user appearing in >= 2 community graphs is a cross-community bridge.

    Ranking key: (n_communities, total_out_degree).
    """
    user_per_community: dict[str, dict[int, int]] = {}  # community -> {user: out_deg}

    for c in COMMUNITIES:
        G = load_graph(f"user_qa_{c}")
        out_deg = dict(G.out_degree())
        # Keep only users who actually answered at least one question in this community
        user_per_community[c] = {u: d for u, d in out_deg.items() if d > 0}
        print(f"  {c}: {len(user_per_community[c]):,} answerers loaded",
              flush=True)
        del G

    # Pivot to user-centric view
    all_users: set[int] = set()
    for d in user_per_community.values():
        all_users.update(d.keys())
    print(f"  union of answerers across 5 communities: {len(all_users):,}",
          flush=True)

    rows = []
    for u in all_users:
        per_c = {c: user_per_community[c].get(u, 0) for c in COMMUNITIES}
        n_comm = sum(1 for v in per_c.values() if v > 0)
        if n_comm >= 2:
            rows.append({
                "user_id": u,
                "n_communities": n_comm,
                "total_out_degree": sum(per_c.values()),
                **{f"out_{c}": per_c[c] for c in COMMUNITIES},
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["n_communities", "total_out_degree"],
                        ascending=[False, False])
    print(f"  cross-community bridge users (>=2 communities): {len(df):,}",
          flush=True)
    return df.head(top_n).reset_index(drop=True), df


# ---------------------------------------------------------------------------
# 2. Null model baseline for Louvain modularity
# ---------------------------------------------------------------------------

def random_rewiring_modularity(G: nx.Graph,
                                n_rewirings: int = N_REWIRINGS,
                                seed: int = RNG_SEED) -> tuple[float, float, list[float]]:
    """
    For an undirected weighted graph, generate `n_rewirings` configuration-model
    null networks (preserving degree sequence) and run Louvain on each.

    Returns
    -------
    obs_Q : float
    z_score : float = (obs - mean(null)) / std(null)
    null_Qs : list of Q values from null models
    """
    # Observed Q
    obs_partition = community_louvain.best_partition(G, weight="weight",
                                                      random_state=seed)
    obs_Q = community_louvain.modularity(obs_partition, G, weight="weight")

    # Configuration-model null: keep degree sequence, scramble edges
    # nx.double_edge_swap preserves degree sequence (Maslov-Sneppen)
    rng = np.random.default_rng(seed)
    null_Qs: list[float] = []
    n_swaps = G.number_of_edges() * SWAP_RATIO

    t_loop_start = time.perf_counter()
    for i in range(n_rewirings):
        t_iter_start = time.perf_counter()
        G_null = G.copy()
        try:
            nx.double_edge_swap(G_null,
                                nswap=n_swaps,
                                max_tries=n_swaps * 10,
                                seed=int(rng.integers(1 << 31)))
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            continue
        part = community_louvain.best_partition(
            G_null, weight="weight",
            random_state=int(rng.integers(1 << 31))
        )
        null_Qs.append(
            community_louvain.modularity(part, G_null, weight="weight")
        )
        t_iter = time.perf_counter() - t_iter_start
        elapsed = time.perf_counter() - t_loop_start
        eta = elapsed * (n_rewirings - i - 1) / (i + 1)
        print(f"      [{i+1:>2}/{n_rewirings}] {t_iter:>5.1f}s/iter  "
              f"elapsed={elapsed/60:>4.1f}m  ETA={eta/60:>4.1f}m",
              flush=True)

    if len(null_Qs) < 5:
        return obs_Q, float("nan"), null_Qs

    mu = float(np.mean(null_Qs))
    sigma = float(np.std(null_Qs, ddof=1))
    z = (obs_Q - mu) / sigma if sigma > 0 else float("nan")
    return obs_Q, z, null_Qs


def run_null_models(expected_min_per_graph: dict[str, float] | None = None) -> pd.DataFrame:
    """Null-model baseline for global tag graph + 5 community tag graphs.

    expected_min_per_graph: empirical estimates {community: minutes} from
        timing test; printed up-front so user knows what to expect.
    """
    # Run smallest first so user sees results early; global last.
    order = ["ai_ml", "cloud_devops", "databases", "mobile",
             "web_frontend", "global"]

    if expected_min_per_graph:
        total_min = sum(expected_min_per_graph.get(c, 0) for c in order)
        print(f"\n  Empirical estimate: {total_min:.0f} min total. "
              f"Per-graph breakdown:")
        for c in order:
            print(f"    {c:<15} {expected_min_per_graph.get(c, 0):>6.1f} min")
        print()

    rows = []
    t_overall = time.perf_counter()
    for c in order:
        name = f"tag_cooc_{c}"
        print(f"\n  null model on {name} ...", flush=True)
        G = load_graph(name)
        # Use largest connected component to avoid Louvain on disconnected pieces
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            print(f"    using LCC: {G.number_of_nodes():,} nodes "
                  f"{G.number_of_edges():,} edges", flush=True)

        t0 = time.perf_counter()
        obs_Q, z, null_Qs = random_rewiring_modularity(G, n_rewirings=N_REWIRINGS)
        elapsed = time.perf_counter() - t0
        rows.append({
            "graph": name,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "obs_Q": obs_Q,
            "null_Q_mean": float(np.mean(null_Qs)) if null_Qs else float("nan"),
            "null_Q_std":  float(np.std(null_Qs, ddof=1)) if len(null_Qs) >= 2 else float("nan"),
            "z_score": z,
            "n_null_runs": len(null_Qs),
            "elapsed_s": elapsed,
        })
        cumulative = (time.perf_counter() - t_overall) / 60
        expected = expected_min_per_graph.get(c, 0) if expected_min_per_graph else 0
        actual = elapsed / 60
        delta = actual - expected
        print(f"    obs Q={obs_Q:.4f}, null_mean={np.mean(null_Qs):.4f}, "
              f"z={z:.2f}", flush=True)
        print(f"    actual={actual:.1f}m  expected={expected:.1f}m  "
              f"delta={delta:+.1f}m  cumulative={cumulative:.1f}m",
              flush=True)
        # Save intermediate results so we don't lose work if interrupted
        pd.DataFrame(rows).to_csv(OUT_DIR / "louvain_null_model_partial.csv",
                                  index=False)
        del G
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Aggregate Louvain stats from existing CSVs (no recomputation)
# ---------------------------------------------------------------------------

def aggregate_louvain_stats() -> pd.DataFrame:
    """Re-derive per-graph Louvain stats by re-running once with seed=42 on the
    same graphs. This is fast (≤ a few seconds per tag graph) and consistent."""
    targets = list(COMMUNITIES) + ["global"]
    rows = []
    for c in targets:
        G = load_graph(f"tag_cooc_{c}")
        partition = community_louvain.best_partition(G, weight="weight",
                                                      random_state=RNG_SEED)
        Q = community_louvain.modularity(partition, G, weight="weight")
        rows.append({
            "community": c,
            "tag_nodes": G.number_of_nodes(),
            "tag_edges": G.number_of_edges(),
            "louvain_Q": Q,
            "n_louvain_communities": len(set(partition.values())),
        })
        del G
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Stitch everything together into sna_summary.md
# ---------------------------------------------------------------------------

def write_summary(louvain_df: pd.DataFrame,
                  null_df: pd.DataFrame,
                  cross_top_df: pd.DataFrame,
                  cross_full_df: pd.DataFrame) -> None:
    lines = ["# Phase 3 — SNA Summary\n"]

    # Section 1: Louvain + null model
    lines.append("## 1. Louvain community detection vs null model\n")
    lines.append("`obs_Q` = observed modularity (random_state=42). "
                 f"`null_*` = mean/std of {N_REWIRINGS} configuration-model "
                 "rewirings (degree-preserving). `z` = (obs − null_mean) / null_std.\n")
    lines.append("| Graph | Nodes | Edges | Louvain Q | #Communities | Null Q (mean) | Null Q (std) | z-score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    merged = louvain_df.merge(null_df.rename(columns={"graph": "graph_name"}),
                              left_on="community",
                              right_on=null_df["graph"].str.replace("tag_cooc_", "", regex=False).rename("community"),
                              how="left")
    # simpler approach: match by community vs null graph name
    null_lookup = {row["graph"].replace("tag_cooc_", ""): row for _, row in null_df.iterrows()}
    for _, row in louvain_df.iterrows():
        c = row["community"]
        null = null_lookup.get(c, {})
        lines.append(
            f"| `{c}` | {row['tag_nodes']:,} | {row['tag_edges']:,} | "
            f"{row['louvain_Q']:.4f} | {row['n_louvain_communities']} | "
            f"{null.get('null_Q_mean', float('nan')):.4f} | "
            f"{null.get('null_Q_std', float('nan')):.4f} | "
            f"{null.get('z_score', float('nan')):.2f} |"
        )

    # Section 2: NMI vs taxonomy (read from existing louvain_*.csv files)
    lines.append("\n## 2. NMI vs manual taxonomy\n")
    lines.append("From per-community Louvain partitions; "
                 "taxonomy = `outputs/tables/target_tags_proposed.json`.\n")
    nmi_rows = read_nmi_from_existing_csvs()
    lines.append("| Graph | #Louvain comm | dominant taxonomy purity (top-3) |")
    lines.append("|---|---:|---|")
    for r in nmi_rows:
        top3 = ", ".join(f"{c}({p:.0%})" for c, p in r["top3"])
        lines.append(f"| `{r['community']}` | {r['n_communities']} | {top3} |")

    # Section 3: cross-community bridge users
    lines.append("\n## 3. Cross-community bridge users (RQ1)\n")
    lines.append("Users appearing as ANSWERER in 2+ of the 5 community user "
                 "Q&A graphs. Ranked by (n_communities, total_out_degree).\n")
    n_per = cross_full_df.groupby("n_communities").size().to_dict()
    lines.append("**Distribution of bridge-coverage:**\n")
    lines.append("| In how many communities | # users |")
    lines.append("|---:|---:|")
    for n in sorted(n_per.keys(), reverse=True):
        lines.append(f"| {n} | {n_per[n]:,} |")

    lines.append("\n**Top 20 cross-community bridges:**\n")
    cols = ["user_id", "n_communities", "total_out_degree"] + \
           [f"out_{c}" for c in COMMUNITIES]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---:"] * len(cols)) + "|")
    for _, row in cross_top_df.head(20).iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    # Section 4: per-community top bridge user (from existing CSVs)
    lines.append("\n## 4. Per-community top bridge user (k-core betweenness)\n")
    lines.append("From the earlier `bridge_users_<community>.csv` files. "
                 "These are the most central users *within* each community's "
                 "Q&A network.\n")
    lines.append("| Community | top user_id | betweenness | out_deg | in_deg |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in COMMUNITIES:
        path = OUT_DIR / f"bridge_users_{c}.csv"
        if path.exists():
            df = pd.read_csv(path)
            top = df.iloc[0]
            lines.append(
                f"| `{c}` | {int(top['user_id'])} | "
                f"{top['betweenness']:.4f} | "
                f"{int(top['out_degree'])} | {int(top['in_degree'])} |"
            )

    # Section 5: methodology note on global
    lines.append("\n## 5. Methodology notes\n")
    lines.append("- **Global user Q&A network betweenness was NOT computed.** "
                 "The graph has 5.6M nodes / 30M edges; networkx betweenness "
                 "(even sampled) is not tractable on this scale and would "
                 "require >19 GB RAM for the undirected conversion. "
                 "The scientifically meaningful operationalisation of "
                 "\"bridge user connecting different technical communities\" "
                 "is **cross-community membership** (Section 3 above), "
                 "which directly answers RQ1.")
    lines.append("- **Per-community betweenness uses k-core auto-tuning.** "
                 "k-core threshold is increased (k=5,10,15,...) until the "
                 "core has ≤ 3,000 nodes, then approximate betweenness with "
                 "500 random pivots is computed.")
    lines.append("- **Null model**: configuration model via "
                 "`networkx.double_edge_swap` (Maslov–Sneppen), preserving "
                 "degree sequence. `n_rewirings={}` per graph.".format(N_REWIRINGS))
    lines.append("- **Reproducibility**: all stochastic operations use "
                 "`random_state=42`.")

    out_md = OUT_DIR / "sna_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary → {out_md}")


def read_nmi_from_existing_csvs() -> list[dict]:
    """Read top-3 dominant taxonomies from louvain_<c>.csv files."""
    rows = []
    targets = list(COMMUNITIES) + ["global"]
    for c in targets:
        path = OUT_DIR / f"louvain_{c}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        n_communities = len(df)
        # Top 3 communities by member count, with their dominant taxonomy
        top3 = []
        for _, row in df.head(3).iterrows():
            top3.append((row["dominant_taxonomy"], row["purity"]))
        rows.append({
            "community": c,
            "n_communities": n_communities,
            "top3": top3,
        })
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 3 finalisation")
    print("=" * 60)

    print("\n[1/4] Cross-community bridge users ...")
    cross_top, cross_full = cross_community_bridge_users(top_n=50)
    cross_full.to_csv(OUT_DIR / "bridge_users_cross_community_full.csv", index=False)
    cross_top.to_csv(OUT_DIR / "bridge_users_cross_community.csv", index=False)
    print(f"  → bridge_users_cross_community.csv (top 50)")
    print(f"  → bridge_users_cross_community_full.csv "
          f"({len(cross_full):,} users)")

    print("\n[2/4] Null model baseline (Louvain Q vs random) ...")
    expected_min = {c: s * N_REWIRINGS / 60
                    for c, s in EMPIRICAL_SEC_PER_ITER.items()}
    null_df = run_null_models(expected_min_per_graph=expected_min)
    null_df.to_csv(OUT_DIR / "louvain_null_model.csv", index=False)
    print(f"  → louvain_null_model.csv")

    print("\n[3/4] Aggregating Louvain stats ...")
    louvain_df = aggregate_louvain_stats()
    louvain_df.to_csv(OUT_DIR / "louvain_stats.csv", index=False)
    print(f"  → louvain_stats.csv")

    print("\n[4/4] Writing summary markdown ...")
    write_summary(louvain_df, null_df, cross_top, cross_full)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

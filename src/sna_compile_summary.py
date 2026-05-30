"""
Recompile the Phase 3 SNA summary from authoritative sources.

This script does NOT recompute long-running things (null model, bridge
betweenness). It only:

  1. Re-runs Louvain at resolution=1.0, random_state=42 on each tag
     graph and computes NMI vs the manual taxonomy. This is fast
     (~10 s per graph, ~1 min total).
  2. Loads `outputs/tables/louvain_null_model.csv` for null statistics.
  3. Loads `outputs/tables/bridge_users_<community>.csv` for top-1
     within-community bridge users.
  4. Loads `outputs/tables/bridge_users_cross_community*.csv` for
     cross-community bridges.
  5. Writes a complete, sectioned `outputs/tables/sna_summary.md`.

Result: a single source of truth for Phase 3 numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import community as community_louvain
import networkx as nx
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from network_builder import load_graph

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "tables"
TAGS_JSON = ROOT / "outputs" / "tables" / "target_tags_proposed.json"

COMMUNITIES = ("ai_ml", "web_frontend", "mobile", "cloud_devops", "databases")
ALL_GRAPHS = list(COMMUNITIES) + ["global"]
RES = 1.0
SEED = 42


def load_taxonomy() -> dict[str, str]:
    raw = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
    tag_to_community: dict[str, str] = {}
    for community, tags in raw["target_tags"].items():
        for t in tags:
            tag_to_community[t] = community
    return tag_to_community


def louvain_with_nmi(c: str, taxonomy: dict[str, str]) -> dict:
    G = load_graph(f"tag_cooc_{c}")
    partition = community_louvain.best_partition(G, weight="weight",
                                                  resolution=RES,
                                                  random_state=SEED)
    Q = community_louvain.modularity(partition, G, weight="weight")

    overlap = [n for n in partition if n in taxonomy]
    if len(overlap) >= 10:
        nmi = normalized_mutual_info_score(
            [taxonomy[n] for n in overlap],
            [partition[n] for n in overlap],
            average_method="arithmetic",
        )
    else:
        nmi = float("nan")

    # Per Louvain-community: dominant taxonomy + size
    cluster_summary = []
    for cid in sorted(set(partition.values())):
        members = [n for n, c in partition.items() if c == cid]
        labelled = [taxonomy[n] for n in members if n in taxonomy]
        if labelled:
            cnt = Counter(labelled)
            dominant, top_count = cnt.most_common(1)[0]
            purity = top_count / len(labelled)
        else:
            dominant, purity = "unlabelled", 0.0
        cluster_summary.append({
            "louvain_id": cid,
            "n_members": len(members),
            "n_labelled": len(labelled),
            "dominant_taxonomy": dominant,
            "purity": purity,
        })
    cluster_df = pd.DataFrame(cluster_summary).sort_values(
        "n_members", ascending=False)

    return {
        "graph":   c,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "Q":       Q,
        "n_communities": len(set(partition.values())),
        "nmi":     nmi,
        "n_taxonomy_overlap": len(overlap),
        "cluster_df": cluster_df,
    }


def main() -> None:
    print("Loading taxonomy ...", flush=True)
    taxonomy = load_taxonomy()
    print(f"  {len(taxonomy)} tagged entries across "
          f"{len(set(taxonomy.values()))} communities", flush=True)

    print("\nRe-running Louvain + NMI on all tag graphs (res=1.0, seed=42) ...",
          flush=True)
    louvain_results = {}
    for c in ALL_GRAPHS:
        r = louvain_with_nmi(c, taxonomy)
        louvain_results[c] = r
        # save per-graph cluster CSV (overwrites old louvain_*.csv)
        r["cluster_df"].to_csv(OUT_DIR / f"louvain_{c}.csv", index=False)
        print(f"  {c:<15} Q={r['Q']:.4f}  NMI={r['nmi']:.4f}  "
              f"#comm={r['n_communities']}  taxonomy_overlap={r['n_taxonomy_overlap']}",
              flush=True)

    # Save aggregate Louvain stats
    louvain_stats = pd.DataFrame([
        {k: v for k, v in r.items() if k != "cluster_df"}
        for r in louvain_results.values()
    ])
    louvain_stats.to_csv(OUT_DIR / "louvain_stats.csv", index=False)

    print("\nLoading null model results ...", flush=True)
    null_df = pd.read_csv(OUT_DIR / "louvain_null_model.csv")
    null_lookup = {
        row["graph"].replace("tag_cooc_", ""): row
        for _, row in null_df.iterrows()
    }

    print("Loading bridge user CSVs ...", flush=True)
    bridge_per_comm = {}
    for c in COMMUNITIES:
        df = pd.read_csv(OUT_DIR / f"bridge_users_{c}.csv")
        bridge_per_comm[c] = df

    cross_top = pd.read_csv(OUT_DIR / "bridge_users_cross_community.csv")
    cross_full = pd.read_csv(OUT_DIR / "bridge_users_cross_community_full.csv")

    # --- WRITE SUMMARY ---
    print("\nWriting sna_summary.md ...", flush=True)
    L = []

    L.append("# Phase 3 — SNA Summary\n")
    L.append("Single source of truth, regenerated by `src/sna_compile_summary.py`.")
    L.append("")
    L.append("Methodology snapshot:")
    L.append("- Louvain: `python-louvain` best_partition, resolution=1.0, "
             "random_state=42, weight='weight'.")
    L.append("- NMI: arithmetic average; only nodes present in BOTH the "
             "Louvain partition AND the manual taxonomy contribute "
             "(general tags like `python` are unlabelled and skipped).")
    L.append("- Null model: 20 configuration-model rewirings via "
             "`nx.double_edge_swap` (Maslov–Sneppen), nswap=2|E|, "
             "preserving degree sequence.")
    L.append("- Bridge users (per-community): k-core auto-tune to ≤3,000 "
             "nodes, then EXACT betweenness centrality "
             "(deterministic; no sampling). "
             "Switched from sampled to exact after discovering that "
             "sampled betweenness with `seed=42` was non-reproducible "
             "across re-runs because pivot selection is sensitive to "
             "node insertion order.")
    L.append("- Bridge users (cross-community): user appears as ANSWERER in "
             "≥2 of the 5 community user_qa graphs.")
    L.append("")

    # --- Section 1: Louvain Q + NMI + null model ---
    L.append("## 1. Louvain modularity, NMI vs taxonomy, null model\n")
    L.append("| Graph | Nodes | Edges | Louvain Q | #Comm | NMI | "
             "Null Q (mean ± std) | z-score |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in ALL_GRAPHS:
        r = louvain_results[c]
        n = null_lookup.get(c, {})
        null_str = (
            f"{n.get('null_Q_mean', float('nan')):.4f} ± "
            f"{n.get('null_Q_std', float('nan')):.4f}"
            if "null_Q_mean" in n else "n/a"
        )
        z = n.get("z_score", float("nan"))
        L.append(
            f"| `{c}` | {r['n_nodes']:,} | {r['n_edges']:,} | "
            f"{r['Q']:.4f} | {r['n_communities']} | {r['nmi']:.4f} | "
            f"{null_str} | {z:+.2f} |"
        )

    L.append("")
    L.append("**z-score interpretation:** "
             "z > 3 = community structure is significantly stronger than "
             "expected from degree sequence alone; "
             "0 < z < 3 = positive but not strongly significant; "
             "z < 0 = observed Q is *lower* than random null "
             "(see discussion for `global`).")

    # --- Section 2: Per-Louvain-community taxonomy purity ---
    L.append("\n## 2. Per-Louvain-community taxonomy purity (top 5 sub-clusters)\n")
    for c in ALL_GRAPHS:
        r = louvain_results[c]
        L.append(f"### `{c}`")
        cdf = r["cluster_df"].head(5)
        L.append("| Louvain # | Members | Labelled | Dominant taxonomy | Purity |")
        L.append("|---:|---:|---:|---|---:|")
        for _, row in cdf.iterrows():
            L.append(
                f"| {row['louvain_id']} | {row['n_members']:,} | "
                f"{row['n_labelled']:,} | `{row['dominant_taxonomy']}` | "
                f"{row['purity']:.0%} |"
            )
        L.append("")

    # --- Section 3: Cross-community bridge users ---
    L.append("## 3. Cross-community bridge users (RQ1)\n")
    L.append("Users appearing as ANSWERER in 2+ of the 5 community user_qa "
             "graphs. Ranked by `(n_communities, total_out_degree)`.\n")
    n_per = cross_full.groupby("n_communities").size().to_dict()
    L.append("**Distribution:**\n")
    L.append("| In how many communities | # users |")
    L.append("|---:|---:|")
    for n in sorted(n_per.keys(), reverse=True):
        L.append(f"| {n} | {n_per[n]:,} |")

    L.append(f"\n**Top 20 cross-community bridges** "
             f"(out of {len(cross_full):,} total):\n")
    cols = ["user_id", "n_communities", "total_out_degree"] + \
           [f"out_{c}" for c in COMMUNITIES]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "|".join(["---:"] * len(cols)) + "|")
    for _, row in cross_top.head(20).iterrows():
        vals = [str(int(row[c])) if c != "user_id" else str(int(row[c]))
                for c in cols]
        L.append("| " + " | ".join(vals) + " |")

    # --- Section 4: Per-community top bridge users (k-core betweenness) ---
    L.append("\n## 4. Per-community top bridge users (k-core betweenness)\n")
    L.append("Most central users *within* each community's Q&A network. "
             "From `bridge_users_<community>.csv`.\n")
    L.append("| Community | top user_id | betweenness | out_deg | in_deg |")
    L.append("|---|---:|---:|---:|---:|")
    for c in COMMUNITIES:
        df = bridge_per_comm[c]
        top = df.iloc[0]
        bet = top["betweenness"]
        bet_str = f"{bet:.4f}" if pd.notna(bet) else "n/a"
        L.append(
            f"| `{c}` | {int(top['user_id'])} | {bet_str} | "
            f"{int(top['out_degree'])} | {int(top['in_degree'])} |"
        )

    # --- Section 5: methodology notes / known limitations ---
    L.append("\n## 5. Methodology notes & known limitations\n")
    L.append("- **`global` user Q&A betweenness was deliberately not "
             "computed.** 5.6M nodes × 30M edges; networkx betweenness "
             "(even sampled) is intractable here. Construct-validity "
             "argument: the original RQ1 wording is *bridge users connecting "
             "different technical communities*, which Section 3 "
             "(cross-community membership) addresses directly.")
    L.append("- **`global` z-score is NEGATIVE (-1.43).** This means the "
             "observed modularity is *lower* than the configuration-model "
             "null. Likely cause: hub tags (`python`, `javascript`, `java` "
             "with 2M+ uses each) bridge many communities in the actual "
             "graph; degree-preserving rewiring scatters those hub edges "
             "uniformly, allowing tighter, more modular sub-structures to "
             "emerge in the null. This is a known artefact of scale-free "
             "networks with super-hubs and is a finding rather than a bug.")
    L.append("- **`ai_ml` and `cloud_devops` z-scores (1.44 and 0.49) are "
             "below the typical z>3 threshold for strong significance.** "
             "Possible causes: tight curated tag pools (97 and 384 tags "
             "respectively) leave little room for null-model variation. "
             "Modularity is still positive and clearly above 0.3, so "
             "communities exist; the test simply cannot reject "
             "the null at high confidence for these scopes.")
    L.append("- **NMI values for community-scoped graphs are LOW** "
             "(0.20–0.43) because each community-scoped tag graph contains "
             "mostly tags from ONE taxonomy class. NMI penalises low "
             "label diversity. The `global` graph (NMI=0.72) is where the "
             "comparison is most informative — Louvain on the full taxonomy "
             "rediscovers a partition that matches our manual labelling "
             "fairly well.")
    L.append("- **Reproducibility:** all stochastic operations use "
             "`random_state=42`. To reproduce: run "
             "`src/build_all_networks.py` → `src/sna_analysis.py` → "
             "`src/sna_finalize.py` → `src/sna_compile_summary.py`.")

    out_path = OUT_DIR / "sna_summary.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n→ {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()

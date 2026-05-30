"""
Build all Phase 2 networks and persist them.

Products
--------
Tag co-occurrence (undirected, weighted; min_weight=5 global, min_weight=3 for smaller scopes):
  - tag_cooc_global.pkl
  - tag_cooc_<community>.pkl          (for each of the 5 communities)
  - tag_cooc_llm_pre_chatgpt.pkl      (LLM sub-pool, < 2022-11-30)
  - tag_cooc_llm_post_chatgpt.pkl     (LLM sub-pool, >= 2022-11-30)

User Q&A (directed, weighted; min_weight=1 to retain bridge users):
  - user_qa_global.pkl
  - user_qa_<community>.pkl
  - user_qa_llm_pre_chatgpt.pkl
  - user_qa_llm_post_chatgpt.pkl

All products are pickle files under data/networks/. A manifest is written to
outputs/tables/network_manifest.md with every graph's size and key metrics.
"""

from __future__ import annotations

import time
from pathlib import Path

import networkx as nx

from network_builder import (
    tag_cooccurrence_graph,
    user_qa_graph,
    save_graph,
    summarize_graph,
    NET_DIR,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "outputs" / "tables" / "network_manifest.md"

COMMUNITIES = ("ai_ml", "web_frontend", "mobile", "cloud_devops", "databases")
CHATGPT_SPLIT = "2022-11-30"


def build_and_save(name: str, builder, **kwargs) -> dict:
    t0 = time.perf_counter()
    G = builder(**kwargs)
    elapsed = time.perf_counter() - t0
    path = save_graph(G, name)
    stats = summarize_graph(G)
    stats["name"] = name
    stats["elapsed_s"] = elapsed
    stats["path"] = str(path.relative_to(ROOT))
    print(f"  [{elapsed:5.1f}s] {name:40s}  "
          f"N={stats['n_nodes']:>8,} E={stats['n_edges']:>10,}", flush=True)
    return stats


def main() -> None:
    records: list[dict] = []

    # ------------------------------------------------------------------
    # Tag co-occurrence
    # ------------------------------------------------------------------
    print("\n=== TAG CO-OCCURRENCE ===", flush=True)

    # Global tag network: uses a higher threshold because 24M Q generates
    # lots of noisy low-count pairs. min_weight=10 still keeps >100k edges
    # and avoids edges like (t1, t2) that co-occurred once in 2009.
    records.append(build_and_save(
        "tag_cooc_global", tag_cooccurrence_graph, min_weight=10,
    ))

    # Per-community tag networks. min_weight=5 matches Ye-Xing-Kapre (2017).
    for c in COMMUNITIES:
        records.append(build_and_save(
            f"tag_cooc_{c}", tag_cooccurrence_graph,
            community=c, min_weight=5,
        ))

    # LLM pre/post ChatGPT — small subset, lower threshold
    records.append(build_and_save(
        "tag_cooc_llm_pre_chatgpt", tag_cooccurrence_graph,
        llm_only=True, to_date=CHATGPT_SPLIT, min_weight=2,
    ))
    records.append(build_and_save(
        "tag_cooc_llm_post_chatgpt", tag_cooccurrence_graph,
        llm_only=True, from_date=CHATGPT_SPLIT, min_weight=2,
    ))

    # ------------------------------------------------------------------
    # User Q&A
    # ------------------------------------------------------------------
    print("\n=== USER Q&A (directed) ===", flush=True)

    records.append(build_and_save(
        "user_qa_global", user_qa_graph, min_weight=1,
    ))

    for c in COMMUNITIES:
        records.append(build_and_save(
            f"user_qa_{c}", user_qa_graph,
            community=c, min_weight=1,
        ))

    records.append(build_and_save(
        "user_qa_llm_pre_chatgpt", user_qa_graph,
        llm_only=True, to_date=CHATGPT_SPLIT, min_weight=1,
    ))
    records.append(build_and_save(
        "user_qa_llm_post_chatgpt", user_qa_graph,
        llm_only=True, from_date=CHATGPT_SPLIT, min_weight=1,
    ))

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    lines = ["# Phase 2 network manifest\n"]
    lines.append("All graphs live in `data/networks/*.pkl` and can be loaded "
                 "via `network_builder.load_graph(name)`.\n")

    lines.append("## Tag co-occurrence graphs (undirected)\n")
    lines.append("| name | nodes | edges | density | max w | mean deg | build time |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in records:
        if not r["name"].startswith("tag_cooc"):
            continue
        lines.append(
            f"| `{r['name']}` | {r['n_nodes']:,} | {r['n_edges']:,} | "
            f"{r['density']:.4f} | {r['max_weight']:,} | "
            f"{r['mean_degree']:.1f} | {r['elapsed_s']:.1f}s |"
        )

    lines.append("\n## User Q&A graphs (directed)\n")
    lines.append("| name | nodes | edges | density | max w | mean deg | build time |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in records:
        if not r["name"].startswith("user_qa"):
            continue
        lines.append(
            f"| `{r['name']}` | {r['n_nodes']:,} | {r['n_edges']:,} | "
            f"{r['density']:.2e} | {r['max_weight']:,} | "
            f"{r['mean_degree']:.2f} | {r['elapsed_s']:.1f}s |"
        )

    MANIFEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nManifest → {MANIFEST}")


if __name__ == "__main__":
    main()

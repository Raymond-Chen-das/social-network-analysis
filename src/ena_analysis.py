"""
Phase 4 — ENA (Epistemic Network Analysis).

Three analyses:
  A. ai_ml pre/post-ChatGPT (97-tag dimension space)
  B. LLM sub-pool pre/post-ChatGPT (20-tag dimension space)  ← RQ2 main result
  C. Cross-community ENA comparison (5 communities, PCA to 2D)

Methodology decisions (confirmed with user 2026-04-26 / -27):
  - Tag vocabulary: fixed per analysis (97 ai_ml or 20 LLM, NOT data-driven
    selection per period — needed for matrix consistency).
  - No frequency-threshold filtering: our tag pools are already curated.
  - Pre/post split: ChatGPT release 2022-11-30 (pre = strict <).
  - Comparison metrics: cosine similarity + Frobenius distance + top
    edge-weight changes (more interpretable than PCA on 2 points).
  - Visualisation: pre/post heatmaps + difference heatmap. PCA only used
    for the 5-community comparison (Analysis C).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

from data_loader import Q_PATH, COMMUNITIES

ROOT = Path(__file__).resolve().parent.parent
TAGS_JSON = ROOT / "outputs" / "tables" / "target_tags_proposed.json"
OUT_TBL = ROOT / "outputs" / "tables"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

CHATGPT_SPLIT = "2022-11-30"

# Same false-positive set as extract_subset.py (Agent 5 review).
# Must match for the ai_ml ENA pool to be consistent with `in_ai_ml` flag.
AI_ML_FALSE_POSITIVES = {
    "oracle-coherence",
    "monad-transformers",
    "class-transformer",
    "stringtokenizer",
}

# LLM sub-pool — same list as in extract_subset.py
LLM_SUBSET = [
    "openai-api", "langchain", "chatgpt-api", "large-language-model",
    "gpt-2", "gpt-3", "gpt-4", "llama", "llama-index", "py-langchain",
    "huggingface-transformers", "huggingface", "stable-diffusion",
    "fine-tuning", "chromadb", "pinecone", "vector-database",
    "openai-whisper", "bert-language-model", "transformer-model",
]


# ---------------------------------------------------------------------------
# Tag pool loaders
# ---------------------------------------------------------------------------

def load_ai_ml_tag_pool() -> list[str]:
    """The 97 curated ai_ml tags (false positives removed)."""
    raw = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
    tags = [t for t in raw["target_tags"]["ai_ml"]
            if t not in AI_ML_FALSE_POSITIVES]
    return sorted(tags)


def load_community_tag_pool(community: str) -> list[str]:
    raw = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
    tags = raw["target_tags"][community]
    if community == "ai_ml":
        tags = [t for t in tags if t not in AI_ML_FALSE_POSITIVES]
    return sorted(tags)


# ---------------------------------------------------------------------------
# ENA matrix construction
# ---------------------------------------------------------------------------

def build_ena_matrix(tag_pool: list[str],
                     community: str | None = None,
                     llm_only: bool = False,
                     from_date: str | None = None,
                     to_date: str | None = None,
                     ) -> tuple[np.ndarray, int]:
    """
    Build a normalised tag co-occurrence matrix for the given filter scope.

    Returns:
        matrix: NxN ndarray, where N = len(tag_pool). matrix[i,j] =
                (# questions containing both tag i and tag j) / n_questions.
                Symmetric, zero diagonal.
        n_questions: # questions in the filter scope.

    Implementation: polars-driven self-join on exploded tags, restricted
    to the tag_pool, grouped by (tag_a, tag_b) for counts.
    """
    # Filter scope
    lf = pl.scan_parquet(Q_PATH).select([
        "Id", "Tags", "CreationDate",
        *[f"in_{c}" for c in COMMUNITIES],
        "in_llm",
    ])
    if community is not None:
        lf = lf.filter(pl.col(f"in_{community}"))
    if llm_only:
        lf = lf.filter(pl.col("in_llm"))
    if from_date is not None:
        lf = lf.filter(pl.col("CreationDate") >= from_date)
    if to_date is not None:
        lf = lf.filter(pl.col("CreationDate") < to_date)

    # Eager partial: count of questions in scope
    n_questions = lf.select(pl.len()).collect().item()
    if n_questions == 0:
        n = len(tag_pool)
        return np.zeros((n, n)), 0

    # Explode tags, restrict to pool
    pool_set = set(tag_pool)
    exploded = (
        lf.select(["Id", "Tags"])
        .explode("Tags")
        .filter(pl.col("Tags").is_in(list(pool_set)))
        .rename({"Tags": "tag"})
    )

    # Self-join on Id → all (tag_a, tag_b) pairs within the same question
    pairs = (
        exploded.join(exploded.rename({"tag": "tag_b"}), on="Id")
        .filter(pl.col("tag") < pl.col("tag_b"))
        .group_by(["tag", "tag_b"])
        .agg(pl.len().alias("count"))
        .collect()
    )

    # Build symmetric matrix
    n = len(tag_pool)
    tag_to_idx = {t: i for i, t in enumerate(tag_pool)}
    M = np.zeros((n, n), dtype=np.float64)
    for row in pairs.iter_rows(named=True):
        i = tag_to_idx[row["tag"]]
        j = tag_to_idx[row["tag_b"]]
        c = row["count"]
        M[i, j] = c
        M[j, i] = c

    # Normalise by n_questions → co-occurrence rate
    M = M / n_questions
    return M, n_questions


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def compare_matrices(M_pre: np.ndarray, M_post: np.ndarray,
                     tag_pool: list[str], top_k: int = 20) -> dict:
    """
    Compare two ENA matrices.

    Returns a dict with:
        cosine_similarity : float in [-1,1] (1 = identical)
        frobenius_distance : float (0 = identical)
        top_gains : top_k edges (i,j) with biggest M_post - M_pre
        top_losses : top_k edges with most negative M_post - M_pre
    """
    # Use upper triangle only (matrices are symmetric, exclude diagonal)
    iu = np.triu_indices_from(M_pre, k=1)
    v_pre = M_pre[iu]
    v_post = M_post[iu]

    # Cosine similarity
    norm_pre = np.linalg.norm(v_pre)
    norm_post = np.linalg.norm(v_post)
    if norm_pre == 0 or norm_post == 0:
        cos_sim = float("nan")
    else:
        cos_sim = float(np.dot(v_pre, v_post) / (norm_pre * norm_post))

    # Frobenius distance
    frob = float(np.linalg.norm(M_post - M_pre))

    # Top gains/losses
    diff = M_post - M_pre
    diff_iu = diff[iu]
    n = len(tag_pool)
    rows = list(iu[0])
    cols = list(iu[1])

    # Sort by diff value
    order = np.argsort(diff_iu)
    losses_idx = order[:top_k]                # most negative
    gains_idx  = order[::-1][:top_k]          # most positive

    def make_edge_list(idx_arr) -> list[dict]:
        out = []
        for k in idx_arr:
            i, j = rows[k], cols[k]
            out.append({
                "tag_a": tag_pool[i],
                "tag_b": tag_pool[j],
                "pre_rate":  float(M_pre[i, j]),
                "post_rate": float(M_post[i, j]),
                "diff":      float(diff[i, j]),
            })
        return out

    return {
        "cosine_similarity": cos_sim,
        "frobenius_distance": frob,
        "top_gains":  make_edge_list(gains_idx),
        "top_losses": make_edge_list(losses_idx),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_three_panel_heatmap(M_pre: np.ndarray, M_post: np.ndarray,
                              tag_pool: list[str],
                              title_prefix: str,
                              output_path: Path,
                              ) -> None:
    """Three-panel heatmap: pre, post, diff (post-pre)."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    diff = M_post - M_pre
    vmax_co = max(M_pre.max(), M_post.max())
    vmax_diff = float(np.abs(diff).max())

    for ax, M, title, vmax, cmap in [
        (axes[0], M_pre,  f"{title_prefix} — pre-ChatGPT",  vmax_co,   "YlOrRd"),
        (axes[1], M_post, f"{title_prefix} — post-ChatGPT", vmax_co,   "YlOrRd"),
        (axes[2], diff,   f"{title_prefix} — diff (post − pre)",
                                                            vmax_diff, "RdBu_r"),
    ]:
        if cmap == "RdBu_r":
            im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax,
                            aspect="auto")
        else:
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(tag_pool)))
        ax.set_yticks(range(len(tag_pool)))
        if len(tag_pool) <= 30:
            ax.set_xticklabels(tag_pool, rotation=90, fontsize=7)
            ax.set_yticklabels(tag_pool, fontsize=7)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path.relative_to(ROOT)}", flush=True)


def plot_top_edges_bar(comp: dict, title: str, output_path: Path,
                       top_n: int = 15) -> None:
    """Bar chart of top gain/loss edges."""
    gains = comp["top_gains"][:top_n]
    losses = comp["top_losses"][:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Gains
    labels_g = [f"{e['tag_a']}-{e['tag_b']}" for e in gains]
    values_g = [e["diff"] for e in gains]
    axes[0].barh(range(len(gains)), values_g, color="#d62728")
    axes[0].set_yticks(range(len(gains)))
    axes[0].set_yticklabels(labels_g, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Δ co-occurrence rate (post − pre)")
    axes[0].set_title(f"{title} — top edges that GAINED weight", fontsize=10)
    axes[0].axvline(0, color="black", linewidth=0.5)

    # Losses
    labels_l = [f"{e['tag_a']}-{e['tag_b']}" for e in losses]
    values_l = [e["diff"] for e in losses]
    axes[1].barh(range(len(losses)), values_l, color="#1f77b4")
    axes[1].set_yticks(range(len(losses)))
    axes[1].set_yticklabels(labels_l, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Δ co-occurrence rate (post − pre)")
    axes[1].set_title(f"{title} — top edges that LOST weight", fontsize=10)
    axes[1].axvline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path.relative_to(ROOT)}", flush=True)


# ---------------------------------------------------------------------------
# Analysis A: ai_ml pre/post
# ---------------------------------------------------------------------------

def analysis_ai_ml_pre_post() -> dict:
    print("\n=== Analysis A: ai_ml pre/post-ChatGPT (97 tags) ===", flush=True)
    pool = load_ai_ml_tag_pool()
    print(f"  tag pool size: {len(pool)}", flush=True)

    t0 = time.perf_counter()
    M_pre, n_pre = build_ena_matrix(pool, community="ai_ml",
                                     to_date=CHATGPT_SPLIT)
    print(f"  pre:  {n_pre:,} questions, "
          f"{(time.perf_counter()-t0):.1f}s", flush=True)

    t0 = time.perf_counter()
    M_post, n_post = build_ena_matrix(pool, community="ai_ml",
                                       from_date=CHATGPT_SPLIT)
    print(f"  post: {n_post:,} questions, "
          f"{(time.perf_counter()-t0):.1f}s", flush=True)

    comp = compare_matrices(M_pre, M_post, pool)
    print(f"  cosine sim: {comp['cosine_similarity']:.4f}", flush=True)
    print(f"  frobenius : {comp['frobenius_distance']:.4f}", flush=True)

    plot_three_panel_heatmap(M_pre, M_post, pool,
                              "ai_ml ENA",
                              OUT_FIG / "ena_ai_ml_pre_post.png")
    plot_top_edges_bar(comp, "ai_ml",
                       OUT_FIG / "ena_ai_ml_top_edges.png")

    # Save raw artefacts
    np.save(OUT_TBL / "ena_ai_ml_pre.npy",  M_pre)
    np.save(OUT_TBL / "ena_ai_ml_post.npy", M_post)
    pd.DataFrame(comp["top_gains"]).to_csv(
        OUT_TBL / "ena_ai_ml_top_gains.csv", index=False)
    pd.DataFrame(comp["top_losses"]).to_csv(
        OUT_TBL / "ena_ai_ml_top_losses.csv", index=False)

    return {
        "n_pre": n_pre, "n_post": n_post,
        "cosine": comp["cosine_similarity"],
        "frobenius": comp["frobenius_distance"],
        "top_gains": comp["top_gains"][:10],
        "top_losses": comp["top_losses"][:10],
    }


# ---------------------------------------------------------------------------
# Analysis B: LLM sub-pool pre/post
# ---------------------------------------------------------------------------

def analysis_llm_pre_post() -> dict:
    print("\n=== Analysis B: LLM sub-pool pre/post-ChatGPT (20 tags) ===",
          flush=True)
    pool = sorted(LLM_SUBSET)
    print(f"  tag pool size: {len(pool)}", flush=True)

    t0 = time.perf_counter()
    M_pre, n_pre = build_ena_matrix(pool, llm_only=True,
                                     to_date=CHATGPT_SPLIT)
    print(f"  pre:  {n_pre:,} questions, "
          f"{(time.perf_counter()-t0):.1f}s", flush=True)

    t0 = time.perf_counter()
    M_post, n_post = build_ena_matrix(pool, llm_only=True,
                                       from_date=CHATGPT_SPLIT)
    print(f"  post: {n_post:,} questions, "
          f"{(time.perf_counter()-t0):.1f}s", flush=True)

    comp = compare_matrices(M_pre, M_post, pool)
    print(f"  cosine sim: {comp['cosine_similarity']:.4f}", flush=True)
    print(f"  frobenius : {comp['frobenius_distance']:.4f}", flush=True)

    plot_three_panel_heatmap(M_pre, M_post, pool,
                              "LLM ENA",
                              OUT_FIG / "ena_llm_pre_post.png")
    plot_top_edges_bar(comp, "LLM",
                       OUT_FIG / "ena_llm_top_edges.png")

    np.save(OUT_TBL / "ena_llm_pre.npy",  M_pre)
    np.save(OUT_TBL / "ena_llm_post.npy", M_post)
    pd.DataFrame(comp["top_gains"]).to_csv(
        OUT_TBL / "ena_llm_top_gains.csv", index=False)
    pd.DataFrame(comp["top_losses"]).to_csv(
        OUT_TBL / "ena_llm_top_losses.csv", index=False)

    return {
        "n_pre": n_pre, "n_post": n_post,
        "cosine": comp["cosine_similarity"],
        "frobenius": comp["frobenius_distance"],
        "top_gains": comp["top_gains"][:10],
        "top_losses": comp["top_losses"][:10],
    }


# ---------------------------------------------------------------------------
# Analysis C: cross-community ENA + PCA
# ---------------------------------------------------------------------------

def analysis_cross_community() -> dict:
    """
    Build ENA matrix per community using the GLOBAL union of all 5 community
    tag pools as the dimension space (so all 5 matrices have identical shape).
    Project the 5 matrices to 2D via PCA.
    """
    print("\n=== Analysis C: cross-community ENA (PCA to 2D) ===",
          flush=True)

    # Union of all 5 community tag pools = global comparison space
    raw = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
    union_tags = sorted({
        t for c in COMMUNITIES for t in raw["target_tags"][c]
    })
    print(f"  union tag space: {len(union_tags)} tags", flush=True)

    matrices = {}
    n_questions = {}
    for c in COMMUNITIES:
        t0 = time.perf_counter()
        M, n_q = build_ena_matrix(union_tags, community=c)
        matrices[c] = M
        n_questions[c] = n_q
        print(f"  {c:<15} {n_q:>10,} Q  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # PCA: each matrix → flatten upper triangle → vector
    iu = np.triu_indices(len(union_tags), k=1)
    vectors = np.array([matrices[c][iu] for c in COMMUNITIES])
    print(f"  PCA input: {vectors.shape}", flush=True)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, c in enumerate(COMMUNITIES):
        ax.scatter(coords[i, 0], coords[i, 1], s=300, c=colours[i],
                   edgecolors="black", linewidths=1.0, label=c)
        ax.annotate(c, (coords[i, 0], coords[i, 1]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("Cross-community ENA — 2D PCA projection")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path = OUT_FIG / "ena_cross_community_pca.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.relative_to(ROOT)}", flush=True)

    # Pairwise cosine similarity matrix between communities
    pairwise = np.zeros((len(COMMUNITIES), len(COMMUNITIES)))
    for i, ci in enumerate(COMMUNITIES):
        for j, cj in enumerate(COMMUNITIES):
            v_i = matrices[ci][iu]
            v_j = matrices[cj][iu]
            denom = np.linalg.norm(v_i) * np.linalg.norm(v_j)
            pairwise[i, j] = (
                np.dot(v_i, v_j) / denom if denom > 0 else float("nan")
            )

    pd.DataFrame(pairwise, index=COMMUNITIES, columns=COMMUNITIES).to_csv(
        OUT_TBL / "ena_cross_community_cosine.csv")

    pd.DataFrame({
        "community": COMMUNITIES,
        "n_questions": [n_questions[c] for c in COMMUNITIES],
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    }).to_csv(OUT_TBL / "ena_cross_community_pca_coords.csv", index=False)

    return {
        "n_questions": n_questions,
        "var_explained": var_explained.tolist(),
        "coords": {c: (float(coords[i, 0]), float(coords[i, 1]))
                   for i, c in enumerate(COMMUNITIES)},
        "pairwise_cosine": pairwise.tolist(),
    }


# ---------------------------------------------------------------------------
# Master summary writer
# ---------------------------------------------------------------------------

def write_summary(A: dict, B: dict, C: dict) -> None:
    L = ["# Phase 4 — ENA Summary",
         "",
         "Single source of truth, generated by `src/ena_analysis.py`.",
         "",
         "## Methodology",
         "",
         "- **Tag vocabulary**: fixed per analysis, not data-driven.",
         "  - Analysis A: 97 curated `ai_ml` tags.",
         "  - Analysis B: 20 curated LLM tags.",
         "  - Analysis C: union of all 5 community tag pools.",
         "- **No frequency filtering**: pools are already curated; "
         "low-frequency LLM tags (gpt-4 = 116 questions) are intentionally "
         "kept because they are the diagnostic for RQ2.",
         "- **Pre/post split**: 2022-11-30 (ChatGPT release; pre = strict <).",
         "- **Cell value**: `M[i,j] = (# questions containing both tag i "
         "and tag j) / total_n_questions`. Symmetric, zero diagonal.",
         "- **Comparison metrics**:",
         "  - Cosine similarity on flattened upper-triangle vectors.",
         "  - Frobenius distance on full matrices.",
         "  - Top edge weight changes (post − pre).",
         "",
         "## Analysis A — `ai_ml` pre/post-ChatGPT (97-tag space)",
         "",
         f"- Pre  ({A['n_pre']:,} questions, before 2022-11-30)",
         f"- Post ({A['n_post']:,} questions, on/after 2022-11-30)",
         f"- **Cosine similarity**: {A['cosine']:.4f}",
         f"- **Frobenius distance**: {A['frobenius']:.6f}",
         "",
         "**Top 10 edges that gained most weight (new co-occurrences "
         "post-ChatGPT):**",
         "",
         "| tag_a | tag_b | pre rate | post rate | Δ |",
         "|---|---|---:|---:|---:|"]
    for e in A["top_gains"]:
        L.append(
            f"| `{e['tag_a']}` | `{e['tag_b']}` | "
            f"{e['pre_rate']:.6f} | {e['post_rate']:.6f} | "
            f"{e['diff']:+.6f} |"
        )
    L.append("")
    L.append("**Top 10 edges that lost most weight:**")
    L.append("")
    L.append("| tag_a | tag_b | pre rate | post rate | Δ |")
    L.append("|---|---|---:|---:|---:|")
    for e in A["top_losses"]:
        L.append(
            f"| `{e['tag_a']}` | `{e['tag_b']}` | "
            f"{e['pre_rate']:.6f} | {e['post_rate']:.6f} | "
            f"{e['diff']:+.6f} |"
        )

    L.append("")
    L.append("## Analysis B — LLM sub-pool pre/post-ChatGPT (20-tag space)")
    L.append("")
    L.append(f"- Pre  ({B['n_pre']:,} questions, before 2022-11-30)")
    L.append(f"- Post ({B['n_post']:,} questions, on/after 2022-11-30)")
    L.append(f"- **Cosine similarity**: {B['cosine']:.4f}")
    L.append(f"- **Frobenius distance**: {B['frobenius']:.6f}")
    L.append("")
    L.append("**Top 10 edges that gained most weight:**")
    L.append("")
    L.append("| tag_a | tag_b | pre rate | post rate | Δ |")
    L.append("|---|---|---:|---:|---:|")
    for e in B["top_gains"]:
        L.append(
            f"| `{e['tag_a']}` | `{e['tag_b']}` | "
            f"{e['pre_rate']:.6f} | {e['post_rate']:.6f} | "
            f"{e['diff']:+.6f} |"
        )
    L.append("")
    L.append("**Top 10 edges that lost most weight:**")
    L.append("")
    L.append("| tag_a | tag_b | pre rate | post rate | Δ |")
    L.append("|---|---|---:|---:|---:|")
    for e in B["top_losses"]:
        L.append(
            f"| `{e['tag_a']}` | `{e['tag_b']}` | "
            f"{e['pre_rate']:.6f} | {e['post_rate']:.6f} | "
            f"{e['diff']:+.6f} |"
        )

    L.append("")
    L.append("## Analysis C — cross-community ENA (PCA to 2D)")
    L.append("")
    L.append(f"PCA variance explained: PC1 = {C['var_explained'][0]:.1%}, "
             f"PC2 = {C['var_explained'][1]:.1%}, "
             f"total = {sum(C['var_explained']):.1%}")
    L.append("")
    L.append("**Community 2D coordinates:**")
    L.append("")
    L.append("| community | n_questions | PC1 | PC2 |")
    L.append("|---|---:|---:|---:|")
    for c in COMMUNITIES:
        x, y = C["coords"][c]
        L.append(f"| `{c}` | {C['n_questions'][c]:,} | "
                 f"{x:+.4f} | {y:+.4f} |")
    L.append("")
    L.append("**Pairwise cosine similarity (community × community):**")
    L.append("")
    L.append("| | " + " | ".join(f"`{c}`" for c in COMMUNITIES) + " |")
    L.append("|---|" + "|".join(["---:"] * len(COMMUNITIES)) + "|")
    pw = np.array(C["pairwise_cosine"])
    for i, ci in enumerate(COMMUNITIES):
        cells = " | ".join(f"{pw[i, j]:.3f}" for j in range(len(COMMUNITIES)))
        L.append(f"| `{ci}` | {cells} |")

    L.append("")
    L.append("## Figures")
    L.append("")
    L.append("- `outputs/figures/ena_ai_ml_pre_post.png` — A heatmap triple")
    L.append("- `outputs/figures/ena_ai_ml_top_edges.png` — A top edges")
    L.append("- `outputs/figures/ena_llm_pre_post.png` — B heatmap triple")
    L.append("- `outputs/figures/ena_llm_top_edges.png` — B top edges")
    L.append("- `outputs/figures/ena_cross_community_pca.png` — C 2D plot")

    out = OUT_TBL / "ena_summary.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n→ {out.relative_to(ROOT)}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60, flush=True)
    print("Phase 4 — ENA analysis", flush=True)
    print("=" * 60, flush=True)
    t_overall = time.perf_counter()

    A = analysis_ai_ml_pre_post()
    B = analysis_llm_pre_post()
    C = analysis_cross_community()
    write_summary(A, B, C)

    print(f"\nTotal: {(time.perf_counter()-t_overall)/60:.1f} min")


if __name__ == "__main__":
    main()

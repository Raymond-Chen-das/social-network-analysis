"""
ENA-style tag co-occurrence network visualization for ai_ml community.

Nodes: 97 ai_ml tags, sized by occurrence frequency, colored by subcategory (4 groups).
Edges: tag co-occurrence (top 30% by weight only), thickness ∝ weight,
       intra-community color (alpha=0.4), inter-community gray (alpha=0.2).
Layout: spring_layout initialized from community quadrant centers so same-group
        nodes start close and remain clustered after force-directed relaxation.

Output: outputs/figures/ena_network_ai_ml.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import networkx as nx
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import Q_PATH
from ena_analysis import load_ai_ml_tag_pool, build_ena_matrix

OUT_FIG = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Complete subcategory mapping covering all 97 ai_ml tags
# ---------------------------------------------------------------------------

SUBCATEGORIES: dict[str, dict] = {
    "Classical ML / Data Science": {
        "color": "#1565C0",   # deep blue
        "center": (-1.0,  1.0),  # top-left quadrant
        "tags": {
            # core data science
            "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
            "statsmodels", "r",
            # ML algorithms
            "machine-learning", "machine-learning-model", "scikit-learn",
            "classification", "regression", "decision-tree", "random-forest",
            "svm", "libsvm", "xgboost", "lightgbm", "catboost",
            # feature / training
            "feature-engineering", "gradient-descent",
            # data science general
            "data-science", "data-science-experience",
            # notebooks
            "jupyter-notebook", "jupyter", "jupyter-lab", "jupyterhub",
            "jupyter-irkernel",
        },
    },
    "Deep Learning": {
        "color": "#C62828",   # deep red
        "center": ( 1.0,  1.0),  # top-right quadrant
        "tags": {
            # frameworks
            "tensorflow", "pytorch", "keras", "caffe", "theano", "mxnet",
            # tensorflow ecosystem
            "tensorflow2.0", "tensorflow2.x", "tensorflow-hub",
            "tensorflow-lite", "tensorflow-serving", "tensorflow-estimator",
            "tensorflow-datasets", "tensorflow-probability",
            "tensorflow-federated", "tensorflow.js", "tensorflowjs-converter",
            # pytorch ecosystem
            "pytorch-lightning", "pytorch-dataloader", "pytorch-geometric",
            # architectures
            "neural-network", "deep-learning", "conv-neural-network",
            "lstm", "recurrent-neural-network", "autoencoder",
            "unet-neural-network",
            # training fundamentals
            "backpropagation", "activation", "activation-function",
            "reinforcement-learning", "openai-gym",
            # interop
            "onnx",
        },
    },
    "NLP / Transformers": {
        "color": "#2E7D32",   # deep green
        "center": (-1.0, -1.0),  # bottom-left quadrant
        "tags": {
            # NLP libraries
            "nlp", "nltk", "spacy", "gensim",
            # HuggingFace
            "huggingface", "huggingface-transformers",
            "huggingface-datasets", "huggingface-tokenizers",
            # models / methods
            "bert-language-model", "transformer-model", "word2vec",
            "word-embedding", "sentence-transformers",
            # tasks
            "tokenize", "sentiment-analysis", "named-entity-recognition",
            "text-classification",
        },
    },
    "LLM / GenAI": {
        "color": "#E65100",   # deep orange
        "center": ( 1.0, -1.0),  # bottom-right quadrant
        "tags": {
            # OpenAI / GPT
            "openai-api", "chatgpt-api", "gpt-2", "gpt-3", "gpt-4",
            "openai-whisper",
            # Open-source LLMs
            "llama", "llama-index",
            # Frameworks
            "langchain", "py-langchain",
            # Large language models general
            "large-language-model", "fine-tuning",
            # Vector / RAG infrastructure
            "chromadb", "pinecone", "vector-database", "weaviate",
            # Generative models
            "stable-diffusion",
            # Cloud AI platform
            "azure-machine-learning-service",
        },
    },
}


def get_tag_meta(tag: str) -> tuple[str, str]:
    """Return (subcategory_name, color) for a tag."""
    for name, meta in SUBCATEGORIES.items():
        if tag in meta["tags"]:
            return name, meta["color"]
    return "Other", "#757575"


# ---------------------------------------------------------------------------
# Node frequency computation
# ---------------------------------------------------------------------------

def compute_tag_frequencies(tag_pool: list[str]) -> dict[str, int]:
    lf = pl.scan_parquet(Q_PATH).filter(pl.col("in_ai_ml")).select(["Id", "Tags"])
    exploded = (
        lf.explode("Tags")
        .filter(pl.col("Tags").is_in(tag_pool))
        .group_by("Tags")
        .agg(pl.len().alias("count"))
        .collect()
    )
    return {row["Tags"]: row["count"] for row in exploded.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_cooccurrence_graph(
    tag_pool: list[str],
    freq: dict[str, int],
    top_pct: float = 0.30,
) -> nx.Graph:
    print("Building ENA co-occurrence matrix (full ai_ml, all time)…")
    M, n_q = build_ena_matrix(tag_pool, community="ai_ml")
    print(f"  n_questions = {n_q:,}")

    G = nx.Graph()
    for tag in tag_pool:
        subcat, color = get_tag_meta(tag)
        G.add_node(tag, freq=freq.get(tag, 1), color=color, subcat=subcat)

    n = len(tag_pool)
    edges_raw = []
    for i in range(n):
        for j in range(i + 1, n):
            w = M[i, j]
            if w > 0:
                edges_raw.append((tag_pool[i], tag_pool[j], w))

    edges_raw.sort(key=lambda x: x[2], reverse=True)
    cutoff = max(1, int(len(edges_raw) * top_pct))
    print(f"  Total edges={len(edges_raw)}, keeping top {top_pct*100:.0f}% → {cutoff}")

    for u, v, w in edges_raw[:cutoff]:
        G.add_edge(u, v, weight=w)

    return G


# ---------------------------------------------------------------------------
# Community bubble layout
# ---------------------------------------------------------------------------

def community_bubble_layout(G: nx.Graph, seed: int = 42) -> dict:
    """
    Position nodes by running spring_layout independently within each
    subcategory subgraph, then translating each cluster to its quadrant.
    This guarantees visual separation of communities regardless of
    inter-community edge density.
    """
    # Collect subcategory membership
    subcat_nodes: dict[str, list] = {name: [] for name in SUBCATEGORIES}
    for node in G.nodes():
        subcat = G.nodes[node]["subcat"]
        if subcat in subcat_nodes:
            subcat_nodes[subcat].append(node)

    # Quadrant centers in final coordinate space — spread wide enough
    quadrant_centers = {name: np.array(meta["center"]) * 2.2
                        for name, meta in SUBCATEGORIES.items()}
    # Bubble radius for local layout scaling
    bubble_radius = 1.3

    rng = np.random.default_rng(seed)
    final_pos: dict = {}

    for subcat_name, nodes in subcat_nodes.items():
        center = quadrant_centers[subcat_name]
        if len(nodes) == 0:
            continue
        if len(nodes) == 1:
            final_pos[nodes[0]] = center
            continue

        # Intra-community subgraph for local layout
        sub = G.subgraph(nodes)
        # Use weights to pull strongly co-occurring nodes closer
        local_pos = nx.spring_layout(
            sub, weight="weight", k=1.2,
            iterations=80, seed=int(rng.integers(0, 99999)),
        )

        # Normalize local positions to [-1, 1] bounding box, then scale
        coords = np.array(list(local_pos.values()))
        mn, mx = coords.min(axis=0), coords.max(axis=0)
        span = mx - mn
        span[span == 0] = 1.0
        normalized = (coords - mn) / span * 2 - 1  # [-1, 1]
        scaled = normalized * bubble_radius

        for node, sc in zip(local_pos.keys(), scaled):
            final_pos[node] = center + sc

    return final_pos


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_network(G: nx.Graph, out_path: Path) -> None:
    print("Computing community bubble layout…")
    pos = community_bubble_layout(G)

    fig, ax = plt.subplots(figsize=(22, 18))
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    node_list = list(G.nodes())
    freqs = np.array([G.nodes[n]["freq"] for n in node_list], dtype=float)
    node_colors = [G.nodes[n]["color"] for n in node_list]

    # Node sizes: 250 (min) → 3500 (max)
    f_min, f_max = freqs.min(), freqs.max()
    if f_max > f_min:
        sizes = 250 + (freqs - f_min) / (f_max - f_min) * 3250
    else:
        sizes = np.full(len(freqs), 600.0)

    # --- Edges ---
    edge_list = list(G.edges(data=True))
    if edge_list:
        weights = np.array([d["weight"] for _, _, d in edge_list])
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            widths = 0.4 + (weights - w_min) / (w_max - w_min) * 5.0
        else:
            widths = np.full(len(weights), 1.0)

        for (u, v, _), lw, w in zip(edge_list, widths, weights):
            cu = G.nodes[u]["color"]
            cv = G.nodes[v]["color"]
            same_comm = (G.nodes[u]["subcat"] == G.nodes[v]["subcat"] and
                         G.nodes[u]["subcat"] != "Other")
            color = cu if same_comm else "#AAAAAA"
            alpha = 0.45 if same_comm else 0.18
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1],
                    color=color, linewidth=lw, alpha=alpha,
                    zorder=1, solid_capstyle="round")

    # --- Nodes ---
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=node_list,
        node_color=node_colors,
        node_size=sizes,
        alpha=0.90,
        linewidths=1.0,
        edgecolors="white",
    )

    # --- Labels ---
    for node, (x, y) in pos.items():
        idx = node_list.index(node)
        sz = sizes[idx]
        fs = 6.0 if sz < 700 else (7.5 if sz < 1500 else 9.0)
        fw = "bold" if sz > 1500 else "normal"
        ax.text(
            x, y + 0.04, node,
            fontsize=fs, ha="center", va="bottom",
            fontweight=fw, color="#111111", zorder=5,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", alpha=0.55, lw=0),
        )

    # --- Legend ---
    handles = []
    for name, meta in SUBCATEGORIES.items():
        handles.append(mpatches.Patch(
            facecolor=meta["color"], label=name,
            edgecolor="white", linewidth=1.5,
        ))
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=12,
        framealpha=0.92,
        title="Knowledge Subcategory",
        title_fontsize=13,
        edgecolor="#CCCCCC",
    )

    # --- Subtitle with stats ---
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    ax.set_title(
        f"ENA Tag Co-occurrence Network — ai_ml Community (2015–2024)\n"
        f"{n_nodes} nodes (tags)  |  {n_edges} edges (top 30% co-occurrence)  |  "
        f"Node size ∝ question frequency  |  Edge thickness ∝ co-occurrence weight",
        fontsize=14, fontweight="bold", pad=20, color="#222222",
    )
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import time
    t0 = time.time()

    tag_pool = load_ai_ml_tag_pool()
    print(f"Tag pool: {len(tag_pool)} tags")

    print("Computing tag frequencies…")
    freq = compute_tag_frequencies(tag_pool)

    # Verify full coverage
    uncovered = [t for t in tag_pool if get_tag_meta(t)[0] == "Other"]
    if uncovered:
        print(f"  WARNING: {len(uncovered)} uncovered tags → {uncovered}")
    else:
        print(f"  All {len(tag_pool)} tags classified into 4 subcategories")

    G = build_cooccurrence_graph(tag_pool, freq, top_pct=0.30)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    out_path = OUT_FIG / "ena_network_ai_ml.png"
    plot_network(G, out_path)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

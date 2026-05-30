"""
Phase 5 — DNA (Dynamic Network Analysis).

Track how the tag co-occurrence knowledge structure of each Stack Overflow
technical community evolves from 2015 to 2024 using sliding-window Louvain.

Methodology (Agent 4 + Agent 1 review, 2026-05-10):
  - Sliding window: 6-month window, 3-month step (50% overlap).
  - Tag pools: community-specific curated pools (identical to Phase 4 ENA).
  - Per window: build tag co-occurrence graph → Louvain (resolution=1.0,
    random_state=42) → record Q, n_communities, density, n_questions.
  - Null model (ai_ml only): N=5 double_edge_swap rewirings per window
    (graphs are small — 97 nodes — so this is cheap).
  - Breakpoint detection: rolling z-score on Q series (|z|>2, window=8).
  - Tech events annotated at 7 exact dates from CLAUDE_CODE_PROMPT.md.
  - Data loading: one parquet scan per community; all window filtering done
    in memory on the collected DataFrame (avoids 36× parquet re-scans).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import community as community_louvain
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import Q_PATH, COMMUNITIES, load_questions

TAGS_JSON = ROOT / "outputs" / "tables" / "target_tags_proposed.json"
OUT_TBL = ROOT / "outputs" / "tables"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TECH_EVENTS = {
    "2015-11-09": "TensorFlow",
    "2017-06-12": "Transformer\nPaper",
    "2018-10-11": "BERT",
    "2020-06-11": "GPT-3",
    "2022-11-30": "ChatGPT",
    "2023-02-24": "LLaMA",
    "2023-03-14": "GPT-4",
}

DNA_START = "2015-01-01"
DNA_END = "2024-03-31"
WINDOW_MONTHS = 6
STEP_MONTHS = 3
NULL_N = 5          # null rewirings per window (ai_ml only)
BASE_SEED = 42

AI_ML_FALSE_POSITIVES = {
    "oracle-coherence", "monad-transformers", "class-transformer", "stringtokenizer",
}

COMMUNITY_COLORS = {
    "ai_ml":        "#1f77b4",
    "web_frontend": "#ff7f0e",
    "mobile":       "#2ca02c",
    "cloud_devops": "#d62728",
    "databases":    "#9467bd",
}


# ---------------------------------------------------------------------------
# Tag pool loader
# ---------------------------------------------------------------------------

def _load_tag_pool(community: str) -> list[str]:
    raw = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
    tags = raw["target_tags"][community]
    if community == "ai_ml":
        tags = [t for t in tags if t not in AI_ML_FALSE_POSITIVES]
    return sorted(tags)


# ---------------------------------------------------------------------------
# Per-window graph builder (operates on already-loaded DataFrame)
# ---------------------------------------------------------------------------

def _build_window_graph(
    df_community: pl.DataFrame,
    tag_pool: list[str],
    window_start: str,
    window_end: str,
) -> tuple[nx.Graph, int]:
    """
    Build tag co-occurrence graph for one time window.

    Args:
        df_community: pre-loaded DataFrame with columns [Id, CreationDate, Tags].
        tag_pool: list of tags that define graph nodes.
        window_start, window_end: ISO-8601 date strings (lower inclusive,
            upper exclusive — matches data_loader convention).

    Returns:
        (G, n_questions) where G is a weighted undirected nx.Graph.
        Edge weight = co-occurrence count / n_questions (same as ENA).
    """
    # Filter to window in memory (fast: polars predicate on string comparison)
    df_w = df_community.filter(
        (pl.col("CreationDate") >= window_start)
        & (pl.col("CreationDate") < window_end)
    )
    n_q = len(df_w)

    G = nx.Graph()
    G.add_nodes_from(tag_pool)

    if n_q == 0:
        return G, 0

    pool_set = set(tag_pool)

    # Polars self-join for co-occurrence pairs
    exploded = (
        df_w.lazy()
        .select(["Id", "Tags"])
        .explode("Tags")
        .filter(pl.col("Tags").is_in(list(pool_set)))
        .rename({"Tags": "tag"})
    )
    try:
        pairs = (
            exploded
            .join(exploded.rename({"tag": "tag_b"}), on="Id")
            .filter(pl.col("tag") < pl.col("tag_b"))
            .group_by(["tag", "tag_b"])
            .agg(pl.len().alias("count"))
            .collect()
        )
    except Exception:
        return G, n_q

    for row in pairs.iter_rows(named=True):
        w = row["count"] / n_q
        G.add_edge(row["tag"], row["tag_b"], weight=w)

    return G, n_q


# ---------------------------------------------------------------------------
# Louvain helper
# ---------------------------------------------------------------------------

def _run_louvain(G: nx.Graph, random_state: int = BASE_SEED) -> tuple[float, int]:
    """Run Louvain and return (modularity_Q, n_communities)."""
    if G.number_of_edges() == 0:
        return 0.0, len(G.nodes())
    partition = community_louvain.best_partition(
        G, weight="weight", resolution=1.0, random_state=random_state
    )
    Q = community_louvain.modularity(partition, G, weight="weight")
    n_comm = len(set(partition.values()))
    return Q, n_comm


# ---------------------------------------------------------------------------
# Null model
# ---------------------------------------------------------------------------

def _null_model_Qs(G: nx.Graph, n: int, base_seed: int) -> list[float]:
    """
    Compute N null-model Q values for a single window graph via
    double_edge_swap (Maslov-Sneppen, degree-preserving).
    """
    if G.number_of_edges() < 5:
        return []
    null_Qs = []
    for k in range(n):
        G_copy = G.copy()
        nswap = max(2 * G_copy.number_of_edges(), 10)
        max_tries = max(10 * nswap, 100)
        try:
            nx.double_edge_swap(
                G_copy, nswap=nswap, max_tries=max_tries,
                seed=base_seed + k
            )
            Q_null, _ = _run_louvain(G_copy, random_state=base_seed + k)
            null_Qs.append(Q_null)
        except Exception:
            pass
    return null_Qs


# ---------------------------------------------------------------------------
# Breakpoint detection (rolling z-score, no external dependency)
# ---------------------------------------------------------------------------

def detect_breakpoints(
    Q_values: list[float],
    window: int = 6,
    threshold: float = 2.0,
) -> list[int]:
    """
    Detect structural breakpoints in Q series via rolling z-score on
    first differences.

    Returns list of indices (into Q_values) where |z| exceeds threshold.
    Nearby breakpoints (within 2 steps) are merged to the highest |z|.
    """
    if len(Q_values) < window + 2:
        return []

    Q = np.array(Q_values, dtype=float)
    dQ = np.diff(Q)  # first differences

    raw_bps: list[tuple[int, float]] = []
    for i in range(window, len(dQ)):
        hist = dQ[i - window: i]
        mu = np.mean(hist)
        sigma = np.std(hist) + 1e-9
        z = (dQ[i] - mu) / sigma
        if abs(z) > threshold:
            # index into Q_values: change at transition i → i+1
            raw_bps.append((i + 1, abs(z)))

    if not raw_bps:
        return []

    # Merge neighbouring breakpoints (keep highest |z|)
    merged: list[int] = []
    raw_bps.sort()
    group_start = 0
    while group_start < len(raw_bps):
        group = [raw_bps[group_start]]
        group_end = group_start + 1
        while group_end < len(raw_bps) and raw_bps[group_end][0] - group[-1][0] <= 2:
            group.append(raw_bps[group_end])
            group_end += 1
        best = max(group, key=lambda x: x[1])
        merged.append(best[0])
        group_start = group_end

    return merged


# ---------------------------------------------------------------------------
# Main per-community DNA runner
# ---------------------------------------------------------------------------

def run_dna_community(
    community: str,
    tag_pool: list[str],
    with_null: bool = False,
) -> pd.DataFrame:
    """
    Run sliding-window DNA for one community.

    Returns a DataFrame with one row per valid time window.
    """
    print(f"\n[{community}] Loading {len(tag_pool)}-tag pool questions from parquet...")
    t0 = time.time()
    df = load_questions(community=community, columns=["Id", "CreationDate", "Tags"],
                        from_date=DNA_START, to_date=DNA_END)
    print(f"[{community}] {len(df):,} questions loaded in {time.time()-t0:.1f}s")

    # Generate window start dates (3-month step, explicit loop for pandas compat)
    window_starts = []
    cur = pd.Timestamp(DNA_START)
    end_ts = pd.Timestamp(DNA_END)
    while cur <= end_ts:
        window_starts.append(cur)
        cur = cur + pd.DateOffset(months=STEP_MONTHS)

    records = []
    for i, ws in enumerate(tqdm(window_starts, desc=f"  windows ({community})")):
        we = ws + pd.DateOffset(months=WINDOW_MONTHS)
        ws_str = ws.strftime("%Y-%m-%d")
        we_str = min(we, pd.Timestamp(DNA_END + "T23:59:59")).strftime("%Y-%m-%d")

        G, n_q = _build_window_graph(df, tag_pool, ws_str, we_str)

        if n_q < 50 or G.number_of_edges() < 5:
            continue

        Q, n_comm = _run_louvain(G, random_state=BASE_SEED)

        null_Qs: list[float] = []
        if with_null:
            null_Qs = _null_model_Qs(G, n=NULL_N, base_seed=BASE_SEED + i * 100)

        rec: dict = {
            "community": community,
            "window_start": ws_str,
            "window_end": we_str,
            "window_mid": (ws + pd.DateOffset(months=WINDOW_MONTHS // 2))
                          .strftime("%Y-%m-%d"),
            "n_questions": n_q,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": round(nx.density(G), 6),
            "modularity_Q": round(Q, 6),
            "n_louvain_communities": n_comm,
            "null_mean_Q": round(float(np.mean(null_Qs)), 6) if null_Qs else None,
            "null_std_Q": round(float(np.std(null_Qs)), 6) if null_Qs else None,
        }
        records.append(rec)

    df_out = pd.DataFrame(records)
    elapsed = time.time() - t0
    print(f"[{community}] Done — {len(df_out)} windows in {elapsed:.1f}s total")
    return df_out


# ---------------------------------------------------------------------------
# Monthly volume (for subplot)
# ---------------------------------------------------------------------------

def compute_monthly_volume(community: str) -> pd.DataFrame:
    """Monthly question count for ai_ml from 2015 onward."""
    df = load_questions(community=community, columns=["CreationDate"],
                        from_date=DNA_START, to_date=DNA_END)
    monthly = (
        df.with_columns(
            pl.col("CreationDate").str.slice(0, 7).alias("month")
        )
        .group_by("month")
        .agg(pl.len().alias("n_questions"))
        .sort("month")
    )
    pdf = monthly.to_pandas()
    pdf["month"] = pd.to_datetime(pdf["month"])
    return pdf.sort_values("month").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_ai_ml_main(df: pd.DataFrame, breakpoint_indices: list[int]) -> None:
    """
    Main DNA figure for ai_ml community (Fig 5 of the final report).

    Top panel: Modularity Q over time with null band + tech events + breakpoints.
    Bottom panel: Monthly question volume.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=False,
    )

    dates = pd.to_datetime(df["window_mid"])
    Q = df["modularity_Q"].values

    # --- Top panel: Q timeline ---
    ax1.plot(dates, Q, color=COMMUNITY_COLORS["ai_ml"], lw=2.0, zorder=3,
             label="Observed Q")

    # Null model band (if available)
    has_null = df["null_mean_Q"].notna().any()
    if has_null:
        null_mean = df["null_mean_Q"].values
        null_std = df["null_std_Q"].fillna(0).values
        ax1.fill_between(dates,
                         null_mean - null_std,
                         null_mean + null_std,
                         alpha=0.25, color="gray", label="Null model ±1σ")
        ax1.plot(dates, null_mean, "--", color="gray", lw=1.0, alpha=0.7)

    # Tech event vertical lines + text using mixed transform (x=data, y=axes)
    tech_ts = {pd.Timestamp(d): label for d, label in TECH_EVENTS.items()}
    event_colors = ["#d62728", "#e377c2", "#8c564b", "#bcbd22",
                    "#17becf", "#ff7f0e", "#9467bd"]

    ax1.set_xlim(pd.Timestamp(DNA_START), pd.Timestamp(DNA_END))
    ax1.set_ylabel("Modularity Q", fontsize=11)
    ax1.set_title("AI/ML Knowledge Network Modularity Over Time (2015–2024)", fontsize=13)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.grid(axis="y", alpha=0.3)

    xtrans = ax1.get_xaxis_transform()  # x=data coords, y=axes [0,1]
    for (event_ts, event_label), ec in zip(tech_ts.items(), event_colors):
        ax1.axvline(event_ts, color=ec, lw=1.2, ls="--", alpha=0.7, zorder=2)
        ax1.text(event_ts, 0.98, event_label, rotation=90, fontsize=6.5,
                 color=ec, va="top", ha="right", alpha=0.9,
                 transform=xtrans)

    # Breakpoints
    for bp_idx in breakpoint_indices:
        if 0 <= bp_idx < len(dates):
            ax1.axvline(dates.iloc[bp_idx], color="green", lw=1.5, ls=":",
                        alpha=0.8, zorder=4)
            ax1.scatter([dates.iloc[bp_idx]], [Q[bp_idx]],
                        color="green", s=60, zorder=5)

    # --- Bottom panel: monthly question volume ---
    try:
        monthly = compute_monthly_volume("ai_ml")
        ax2.bar(monthly["month"], monthly["n_questions"],
                width=20, color=COMMUNITY_COLORS["ai_ml"], alpha=0.6)
        for event_ts in tech_ts:
            ax2.axvline(event_ts, color="red", lw=0.8, ls="--", alpha=0.4)
        ax2.set_ylabel("Questions/Month", fontsize=10)
        ax2.set_xlabel("Year", fontsize=11)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
        ))
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_xlim(pd.Timestamp(DNA_START), pd.Timestamp(DNA_END))
    except Exception as e:
        ax2.text(0.5, 0.5, f"Volume plot unavailable:\n{e}",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    out = OUT_FIG / "dna_ai_ml_main.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_all_communities(dfs: dict[str, pd.DataFrame]) -> None:
    """Modularity Q for all 5 communities on one figure."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for community, df in dfs.items():
        if df.empty:
            continue
        dates = pd.to_datetime(df["window_mid"])
        ax.plot(dates, df["modularity_Q"],
                color=COMMUNITY_COLORS[community], lw=1.8,
                label=community.replace("_", " ").title())

    tech_ts = {pd.Timestamp(d): label for d, label in TECH_EVENTS.items()}
    event_colors = ["#d62728", "#e377c2", "#8c564b", "#bcbd22",
                    "#17becf", "#ff7f0e", "#9467bd"]
    for (event_ts, event_label), ec in zip(tech_ts.items(), event_colors):
        ax.axvline(event_ts, color=ec, lw=1.0, ls="--", alpha=0.5)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Modularity Q (Louvain, resolution=1.0)", fontsize=11)
    ax.set_title("Community Modularity Evolution Across 5 SO Communities (2015–2024)",
                 fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(pd.Timestamp(DNA_START), pd.Timestamp(DNA_END))

    plt.tight_layout()
    out = OUT_FIG / "dna_all_communities.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_n_communities(dfs: dict[str, pd.DataFrame]) -> None:
    """Number of Louvain sub-communities per window for ai_ml."""
    df = dfs.get("ai_ml")
    if df is None or df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    dates = pd.to_datetime(df["window_mid"])
    ax.step(dates, df["n_louvain_communities"], where="mid",
            color=COMMUNITY_COLORS["ai_ml"], lw=1.8)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("# Louvain Communities", fontsize=11)
    ax.set_title("AI/ML Sub-Community Count Over Time (Louvain, resolution=1.0)",
                 fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(pd.Timestamp(DNA_START), pd.Timestamp(DNA_END))

    tech_ts = {pd.Timestamp(d): label for d, label in TECH_EVENTS.items()}
    event_colors = ["#d62728", "#e377c2", "#8c564b", "#bcbd22",
                    "#17becf", "#ff7f0e", "#9467bd"]
    xtrans = ax.get_xaxis_transform()
    for (event_ts, event_label), ec in zip(tech_ts.items(), event_colors):
        ax.axvline(event_ts, color=ec, lw=1.0, ls="--", alpha=0.6)
        ax.text(event_ts, 0.98, event_label, rotation=90, fontsize=6.5,
                color=ec, va="top", ha="right", alpha=0.85,
                transform=xtrans)

    plt.tight_layout()
    out = OUT_FIG / "dna_ai_ml_n_communities.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _tech_event_alignment(df: pd.DataFrame) -> list[dict]:
    """
    For each tech event, find the two nearest windows and compute ΔQ.
    'Before' = last window ending before event; 'after' = first window
    starting after event.
    """
    rows = []
    dates_mid = pd.to_datetime(df["window_mid"])
    for event_date_str, event_label in TECH_EVENTS.items():
        event_ts = pd.Timestamp(event_date_str)
        before = df[dates_mid < event_ts]
        after = df[dates_mid >= event_ts]
        if before.empty or after.empty:
            rows.append({
                "event": event_label.replace("\n", " "),
                "date": event_date_str,
                "Q_before": None,
                "Q_after": None,
                "delta_Q": None,
            })
            continue
        Q_before = float(before.iloc[-1]["modularity_Q"])
        Q_after = float(after.iloc[0]["modularity_Q"])
        rows.append({
            "event": event_label.replace("\n", " "),
            "date": event_date_str,
            "Q_before": round(Q_before, 4),
            "Q_after": round(Q_after, 4),
            "delta_Q": round(Q_after - Q_before, 4),
        })
    return rows


def write_dna_summary(
    dfs: dict[str, pd.DataFrame],
    breakpoint_indices: list[int],
) -> None:
    ai_ml_df = dfs.get("ai_ml", pd.DataFrame())

    lines: list[str] = []
    lines.append("# Phase 5 — DNA Summary\n")
    lines.append("Single source of truth, generated by `src/dna_analysis.py`.\n")

    lines.append("## Methodology\n")
    lines.append(f"- **Sliding window**: {WINDOW_MONTHS}-month window, "
                 f"{STEP_MONTHS}-month step (50% overlap).\n")
    lines.append(f"- **Date range**: {DNA_START} to {DNA_END}.\n")
    lines.append("- **Tag pools**: community-specific curated pools "
                 "(identical to Phase 4 ENA; 97/363/41/384/23 tags).\n")
    lines.append("- **Louvain**: resolution=1.0, random_state=42. "
                 "Metric: modularity Q.\n")
    lines.append(f"- **Null model** (ai_ml only): N={NULL_N} "
                 "double_edge_swap rewirings per window.\n")
    lines.append("- **Breakpoint detection**: rolling z-score on first "
                 "differences of Q (|z|>2, window=6).\n")
    lines.append("- **Tech events**: 7 annotated events from CLAUDE_CODE_PROMPT.md.\n")

    lines.append("\n## Window Coverage\n")
    lines.append("| community | n_windows | n_questions_total | Q_mean | Q_min | Q_max |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for comm in COMMUNITIES:
        df = dfs.get(comm, pd.DataFrame())
        if df.empty:
            lines.append(f"| `{comm}` | 0 | — | — | — | — |\n")
            continue
        n_w = len(df)
        n_q = int(df["n_questions"].sum())
        q_mean = round(df["modularity_Q"].mean(), 4)
        q_min = round(df["modularity_Q"].min(), 4)
        q_max = round(df["modularity_Q"].max(), 4)
        lines.append(f"| `{comm}` | {n_w} | {n_q:,} | {q_mean} | {q_min} | {q_max} |\n")

    if not ai_ml_df.empty:
        lines.append("\n## AI/ML Temporal Snapshot (every 4th window)\n")
        lines.append("| window_start | window_end | n_q | Q | n_comm | null_Q | z |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|\n")
        step = max(1, len(ai_ml_df) // 20)
        for _, row in ai_ml_df.iloc[::step].iterrows():
            null_q = f"{row['null_mean_Q']:.4f}" if pd.notna(row["null_mean_Q"]) else "—"
            if (pd.notna(row["null_mean_Q"])
                    and pd.notna(row["null_std_Q"])
                    and row["null_std_Q"] > 0):
                z = (row["modularity_Q"] - row["null_mean_Q"]) / row["null_std_Q"]
                z_str = f"{z:.1f}"
            else:
                z_str = "—"
            lines.append(
                f"| `{row['window_start']}` | `{row['window_end']}` "
                f"| {int(row['n_questions']):,} "
                f"| {row['modularity_Q']:.4f} "
                f"| {int(row['n_louvain_communities'])} "
                f"| {null_q} | {z_str} |\n"
            )

    if not ai_ml_df.empty and breakpoint_indices:
        lines.append("\n## Detected Structural Breakpoints (ai_ml)\n")
        lines.append(
            "Breakpoints detected by rolling z-score on ΔQ "
            "(|z|>2, rolling window=6 steps).\n"
        )
        lines.append("| idx | window_start | Q | ΔQ from prev |\n")
        lines.append("|---|---|---:|---:|\n")
        Q_vals = ai_ml_df["modularity_Q"].values
        for bp in breakpoint_indices:
            if 0 <= bp < len(ai_ml_df):
                row = ai_ml_df.iloc[bp]
                dq = (Q_vals[bp] - Q_vals[bp - 1]) if bp > 0 else float("nan")
                lines.append(
                    f"| {bp} | `{row['window_start']}` "
                    f"| {row['modularity_Q']:.4f} "
                    f"| {dq:+.4f} |\n"
                )

    if not ai_ml_df.empty:
        lines.append("\n## Tech Event Alignment (ai_ml)\n")
        lines.append(
            "Q_before = last window ending before event; "
            "Q_after = first window starting after event.\n"
        )
        lines.append("| event | date | Q_before | Q_after | ΔQ |\n")
        lines.append("|---|---|---:|---:|---:|\n")
        for ev in _tech_event_alignment(ai_ml_df):
            qb = f"{ev['Q_before']:.4f}" if ev["Q_before"] is not None else "—"
            qa = f"{ev['Q_after']:.4f}" if ev["Q_after"] is not None else "—"
            dq = (f"{ev['delta_Q']:+.4f}"
                  if ev["delta_Q"] is not None else "—")
            lines.append(
                f"| {ev['event']} | {ev['date']} | {qb} | {qa} | {dq} |\n"
            )

    lines.append("\n## Figures\n")
    lines.append("- `outputs/figures/dna_ai_ml_main.png` — main DNA figure "
                 "(Q + null band + events + breakpoints)\n")
    lines.append("- `outputs/figures/dna_all_communities.png` — Q for all 5 communities\n")
    lines.append("- `outputs/figures/dna_ai_ml_n_communities.png` — "
                 "Louvain sub-community count (ai_ml)\n")

    summary_path = OUT_TBL / "dna_summary.md"
    summary_path.write_text("".join(lines), encoding="utf-8")
    print(f"Saved {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    print("=" * 60)
    print("Phase 5 — DNA Analysis")
    print(f"Window: {WINDOW_MONTHS}m, Step: {STEP_MONTHS}m, "
          f"Range: {DNA_START} – {DNA_END}")
    print(f"Null rewirings per window (ai_ml): {NULL_N}")
    print("=" * 60)

    # Load tag pools
    tag_pools = {c: _load_tag_pool(c) for c in COMMUNITIES}
    for c, pool in tag_pools.items():
        print(f"  {c}: {len(pool)} tags")

    # Run DNA for all communities
    dfs: dict[str, pd.DataFrame] = {}
    for community in COMMUNITIES:
        df = run_dna_community(
            community=community,
            tag_pool=tag_pools[community],
            with_null=(community == "ai_ml"),
        )
        dfs[community] = df

    # Save raw timeline CSV
    csv_path = OUT_TBL / "dna_timeline.csv"
    all_df = pd.concat(list(dfs.values()), ignore_index=True)
    all_df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path} ({len(all_df)} rows)")

    # Detect breakpoints for ai_ml
    ai_ml_df = dfs.get("ai_ml", pd.DataFrame())
    breakpoint_indices: list[int] = []
    if not ai_ml_df.empty:
        Q_vals = ai_ml_df["modularity_Q"].tolist()
        breakpoint_indices = detect_breakpoints(Q_vals, window=6, threshold=2.0)
        print(f"\nai_ml breakpoints (indices): {breakpoint_indices}")
        bp_windows = [ai_ml_df.iloc[i]["window_start"]
                      for i in breakpoint_indices if i < len(ai_ml_df)]
        print(f"ai_ml breakpoints (dates):   {bp_windows}")

    # Generate figures
    print("\nGenerating figures...")
    if not ai_ml_df.empty:
        plot_ai_ml_main(ai_ml_df, breakpoint_indices)
        plot_n_communities(dfs)
    plot_all_communities(dfs)

    # Write summary
    write_dna_summary(dfs, breakpoint_indices)

    total = time.time() - t_start
    print(f"\nPhase 5 complete — {total:.1f}s total")


if __name__ == "__main__":
    main()

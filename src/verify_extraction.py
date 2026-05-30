"""
Verify the Parquet extraction:
  - counts match full-scan totals
  - community flags are consistent
  - LLM subset pre/post ChatGPT split looks plausible
  - Tags list round-trip preserved
  - date range intact
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "data" / "processed" / "questions.parquet"
A = ROOT / "data" / "processed" / "answers.parquet"
OUT = ROOT / "outputs" / "tables" / "extraction_verification.md"

COMMUNITIES = ("ai_ml", "web_frontend", "mobile", "cloud_devops", "databases")


def main() -> None:
    lf_q = pl.scan_parquet(Q)
    lf_a = pl.scan_parquet(A)

    # 1. Basic counts & date range
    stats_q = lf_q.select([
        pl.len().alias("n_rows"),
        pl.col("CreationDate").min().alias("min_date"),
        pl.col("CreationDate").max().alias("max_date"),
        pl.col("OwnerUserId").null_count().alias("null_owners"),
    ]).collect().row(0, named=True)

    stats_a = lf_a.select([
        pl.len().alias("n_rows"),
        pl.col("CreationDate").min().alias("min_date"),
        pl.col("CreationDate").max().alias("max_date"),
        pl.col("OwnerUserId").null_count().alias("null_owners"),
    ]).collect().row(0, named=True)

    # 2. Community flag distribution
    community_counts = lf_q.select([
        pl.col(f"in_{c}").sum().alias(c) for c in COMMUNITIES
    ] + [pl.col("in_llm").sum().alias("llm")]).collect().row(0, named=True)

    # 3. LLM pre/post ChatGPT split
    pre = lf_q.filter(pl.col("in_llm") & (pl.col("CreationDate") < "2022-11-30")).select(pl.len()).collect().item()
    post = lf_q.filter(pl.col("in_llm") & (pl.col("CreationDate") >= "2022-11-30")).select(pl.len()).collect().item()

    # 4. Tags round-trip (check 5 random rows)
    sample = lf_q.head(5).select(["Id", "Tags", "Title"]).collect()

    # 5. Community × year heatmap (questions only)
    year_expr = pl.col("CreationDate").str.slice(0, 4).alias("year")
    year_community = lf_q.with_columns(year_expr).group_by("year").agg([
        pl.col(f"in_{c}").sum().alias(c) for c in COMMUNITIES
    ] + [pl.col("in_llm").sum().alias("llm"),
         pl.len().alias("total")]).sort("year").collect()

    # 6. At least one community flag per row
    multi = lf_q.with_columns(
        pl.sum_horizontal(*[pl.col(f"in_{c}").cast(pl.Int32)
                            for c in COMMUNITIES]).alias("n_communities")
    ).group_by("n_communities").len().sort("n_communities").collect()

    lines = ["# Extraction verification\n"]
    lines.append("## Counts & coverage\n")
    lines.append(f"- Questions: **{stats_q['n_rows']:,}** rows "
                 f"(expected 24,101,803)")
    lines.append(f"- Answers:   **{stats_a['n_rows']:,}** rows "
                 f"(expected 35,603,624)")
    lines.append(f"- Questions date range: {stats_q['min_date']} → "
                 f"{stats_q['max_date']}")
    lines.append(f"- Answers   date range: {stats_a['min_date']} → "
                 f"{stats_a['max_date']}")
    lines.append(f"- Null OwnerUserId Q: {stats_q['null_owners']:,} "
                 f"({stats_q['null_owners']/stats_q['n_rows']*100:.2f}%)")
    lines.append(f"- Null OwnerUserId A: {stats_a['null_owners']:,} "
                 f"({stats_a['null_owners']/stats_a['n_rows']*100:.2f}%)")
    lines.append("")

    lines.append("## Community flag distribution (questions only)\n")
    lines.append("| Community | Questions | % of Q |")
    lines.append("|---|---:|---:|")
    for c in COMMUNITIES:
        n = community_counts[c]
        lines.append(f"| `{c}` | {n:,} | {n/stats_q['n_rows']*100:.2f}% |")
    lines.append(f"| `llm` (subset of ai_ml) | "
                 f"{community_counts['llm']:,} | "
                 f"{community_counts['llm']/stats_q['n_rows']*100:.2f}% |")
    lines.append("")

    lines.append("## LLM subset pre/post ChatGPT (split 2022-11-30)\n")
    lines.append(f"- Pre  (< 2022-11-30): **{pre:,}** LLM-tagged questions")
    lines.append(f"- Post (>= 2022-11-30): **{post:,}** LLM-tagged questions")
    ratio = post / max(pre, 1)
    lines.append(f"- Post/Pre ratio: **{ratio:.2f}x**")
    lines.append("")

    lines.append("## Community membership overlap\n")
    lines.append("(how many of the 5 community flags a question has)\n")
    lines.append("| n_communities | questions |")
    lines.append("|---:|---:|")
    for row in multi.iter_rows(named=True):
        lines.append(f"| {row['n_communities']} | {row['len']:,} |")
    lines.append("")

    lines.append("## Community × year (questions)\n")
    lines.append("| year | total | ai_ml | llm | web_frontend | mobile | "
                 "cloud_devops | databases |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in year_community.iter_rows(named=True):
        lines.append(f"| {row['year']} | {row['total']:,} | "
                     f"{row['ai_ml']:,} | {row['llm']:,} | "
                     f"{row['web_frontend']:,} | {row['mobile']:,} | "
                     f"{row['cloud_devops']:,} | {row['databases']:,} |")
    lines.append("")

    lines.append("## Tag round-trip sample (first 5 rows)\n")
    for row in sample.iter_rows(named=True):
        tags = ", ".join(f"`{t}`" for t in row["Tags"])
        title = (row['Title'] or '')[:80]
        lines.append(f"- Id={row['Id']}  Tags=[{tags}]")
        lines.append(f"  Title: *{title}*")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written → {OUT}")

    # Console summary
    print("\n--- summary ---")
    print(f"Q rows: {stats_q['n_rows']:,} (expected 24,101,803)")
    print(f"A rows: {stats_a['n_rows']:,} (expected 35,603,624)")
    print(f"LLM pre/post ChatGPT: {pre:,} / {post:,}  ratio {ratio:.2f}x")
    print("Community flag sums:")
    for c in COMMUNITIES:
        print(f"  {c:15s}  {community_counts[c]:>12,}")
    print(f"  llm            {community_counts['llm']:>12,}")


if __name__ == "__main__":
    main()

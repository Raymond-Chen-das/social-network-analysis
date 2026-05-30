"""
Convenience loaders for the processed Parquet dumps produced by
`extract_subset.py`. All phases downstream should use these instead of
touching Posts.xml directly.

Backend: polars.LazyFrame for predicate-pushdown filtering, then .collect().
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
Q_PATH = ROOT / "data" / "processed" / "questions.parquet"
A_PATH = ROOT / "data" / "processed" / "answers.parquet"

COMMUNITIES = ("ai_ml", "web_frontend", "mobile", "cloud_devops", "databases")


def _check_paths() -> None:
    for p in (Q_PATH, A_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run `src/extract_subset.py` first.")


def load_questions(
    community: str | None = None,
    llm_only: bool = False,
    from_date: str | None = None,
    to_date: str | None = None,
    columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    """
    Load questions with optional filters.

    Args:
        community: one of COMMUNITIES. Keeps rows where in_<community> is True.
        llm_only: if True, keep only rows with in_llm=True.
        from_date, to_date: ISO-8601 strings; inclusive lower bound, exclusive
                            upper bound, compared lexicographically on the raw
                            CreationDate strings (works because ISO-8601 sorts
                            lexicographically).
        columns: if given, project only these columns.

    Returns:
        Eager polars DataFrame.
    """
    _check_paths()
    lf = pl.scan_parquet(Q_PATH)

    if community is not None:
        if community not in COMMUNITIES:
            raise ValueError(f"community must be one of {COMMUNITIES}")
        lf = lf.filter(pl.col(f"in_{community}"))

    if llm_only:
        lf = lf.filter(pl.col("in_llm"))

    if from_date is not None:
        lf = lf.filter(pl.col("CreationDate") >= from_date)
    if to_date is not None:
        lf = lf.filter(pl.col("CreationDate") < to_date)

    if columns is not None:
        lf = lf.select(list(columns))

    return lf.collect()


def load_answers_for_questions(question_ids: Iterable[int]) -> pl.DataFrame:
    """Load all answers whose ParentId is in the given set."""
    _check_paths()
    ids = list(question_ids)
    return (
        pl.scan_parquet(A_PATH)
        .filter(pl.col("ParentId").is_in(ids))
        .collect()
    )


def load_answers(
    from_date: str | None = None,
    to_date: str | None = None,
    columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    _check_paths()
    lf = pl.scan_parquet(A_PATH)
    if from_date is not None:
        lf = lf.filter(pl.col("CreationDate") >= from_date)
    if to_date is not None:
        lf = lf.filter(pl.col("CreationDate") < to_date)
    if columns is not None:
        lf = lf.select(list(columns))
    return lf.collect()


def parse_creation_date(df: pl.DataFrame,
                        col: str = "CreationDate") -> pl.DataFrame:
    """Cast ISO-8601 string column to Datetime. Call only when you actually
    need datetime ops — string comparisons are faster for range filtering."""
    return df.with_columns(
        pl.col(col).str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f",
                                    strict=False)
    )

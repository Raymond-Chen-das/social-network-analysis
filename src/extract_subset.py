"""
Posts.xml → Parquet extraction (single SAX pass).

Outputs:
  data/processed/questions.parquet   (~24.1M rows, Q metadata + community flags)
  data/processed/answers.parquet     (~35.6M rows, A metadata for user-network)
  outputs/tables/extraction_summary.json

Design:
  - `xml.parsers.expat` for O(1) memory on the 96 GB file.
  - `pyarrow.parquet.ParquetWriter` writes 100k-row row-groups incrementally,
    so peak RAM stays in the hundreds of MB.
  - zstd compression (fast decode, strong ratio for text-heavy columns).

Schema decisions:
  - `CreationDate` stored as ISO-8601 string. Downstream casts via
    Polars/pandas — parsing 60M timestamps here adds ~10 min for no gain.
  - 5 Boolean `in_<community>` columns + `in_llm` derived at write time
    from the final curated tag sets (so downstream filters cost ~0).
"""

from __future__ import annotations

import json
import sys
import time
import xml.parsers.expat
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
POSTS_XML = ROOT / "Posts.xml"
TAGS_JSON = ROOT / "outputs" / "tables" / "target_tags_proposed.json"
OUT_Q = ROOT / "data" / "processed" / "questions.parquet"
OUT_A = ROOT / "data" / "processed" / "answers.parquet"
OUT_SUMMARY = ROOT / "outputs" / "tables" / "extraction_summary.json"

# --- load tag membership -------------------------------------------------
target = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
TARGET_TAGS: dict[str, list[str]] = target["target_tags"]

# Remove known false positives (Agent 5 review).
AI_ML_FALSE_POSITIVES = {
    "oracle-coherence",       # Oracle in-memory data grid — not AI
    "monad-transformers",     # Haskell — matched "transformer" pattern
    "class-transformer",      # TypeScript DI lib
    "stringtokenizer",        # Java utility class
}
if "ai_ml" in TARGET_TAGS:
    TARGET_TAGS["ai_ml"] = [t for t in TARGET_TAGS["ai_ml"]
                            if t not in AI_ML_FALSE_POSITIVES]

COMMUNITIES: list[str] = list(TARGET_TAGS.keys())
COMMUNITY_SETS: dict[str, set[str]] = {
    c: set(ts) for c, ts in TARGET_TAGS.items()
}

# Hand-curated LLM / generative-AI subset for RQ2 (pre/post-ChatGPT ENA).
LLM_SUBSET: set[str] = {
    "openai-api", "langchain", "chatgpt-api", "large-language-model",
    "gpt-2", "gpt-3", "gpt-4", "llama", "llama-index", "py-langchain",
    "huggingface-transformers", "huggingface", "stable-diffusion",
    "fine-tuning", "chromadb", "pinecone", "vector-database",
    "openai-whisper", "bert-language-model", "transformer-model",
}

# --- batch buffers -------------------------------------------------------
BATCH_SIZE = 100_000

Q_SCHEMA = pa.schema([
    ("Id", pa.int64()),
    ("CreationDate", pa.string()),
    ("OwnerUserId", pa.int32()),
    ("Score", pa.int32()),
    ("ViewCount", pa.int32()),
    ("AnswerCount", pa.int32()),
    ("Tags", pa.list_(pa.string())),
    ("Title", pa.string()),
    ("AcceptedAnswerId", pa.int64()),
    *[(f"in_{c}", pa.bool_()) for c in COMMUNITIES],
    ("in_llm", pa.bool_()),
])
A_SCHEMA = pa.schema([
    ("Id", pa.int64()),
    ("ParentId", pa.int64()),
    ("CreationDate", pa.string()),
    ("OwnerUserId", pa.int32()),
    ("Score", pa.int32()),
])

q_buf: dict[str, list] = {f.name: [] for f in Q_SCHEMA}
a_buf: dict[str, list] = {f.name: [] for f in A_SCHEMA}


def _to_int(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def parse_pipe_tags(s: str | None) -> list[str]:
    if not s:
        return []
    return [t for t in s.split("|") if t]


def flush_q(writer: pq.ParquetWriter) -> None:
    if not q_buf["Id"]:
        return
    writer.write_table(pa.table(q_buf, schema=Q_SCHEMA))
    for v in q_buf.values():
        v.clear()


def flush_a(writer: pq.ParquetWriter) -> None:
    if not a_buf["Id"]:
        return
    writer.write_table(pa.table(a_buf, schema=A_SCHEMA))
    for v in a_buf.values():
        v.clear()


def main() -> int:
    if not POSTS_XML.exists():
        print(f"ERROR: {POSTS_XML} not found", file=sys.stderr)
        return 1
    OUT_Q.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("extract_subset — Posts.xml → Parquet", flush=True)
    print(f"  Source:      {POSTS_XML}", flush=True)
    print(f"  Q output:    {OUT_Q}", flush=True)
    print(f"  A output:    {OUT_A}", flush=True)
    print(f"  Communities: {COMMUNITIES}", flush=True)
    print(f"  LLM_SUBSET:  {len(LLM_SUBSET)} tags", flush=True)
    print("=" * 72, flush=True)

    n_total = n_q = n_a = n_other = 0
    t0 = time.perf_counter()
    last_report = t0
    file_size = POSTS_XML.stat().st_size
    bytes_read = 0

    qw = pq.ParquetWriter(OUT_Q, Q_SCHEMA, compression="zstd")
    aw = pq.ParquetWriter(OUT_A, A_SCHEMA, compression="zstd")

    try:
        def on_start(name: str, attrs: dict) -> None:
            nonlocal n_total, n_q, n_a, n_other
            if name != "row":
                return
            n_total += 1
            pt = attrs.get("PostTypeId", "")
            if pt == "1":
                tags = parse_pipe_tags(attrs.get("Tags"))
                tag_set = set(tags)
                q_buf["Id"].append(_to_int(attrs.get("Id")))
                q_buf["CreationDate"].append(attrs.get("CreationDate"))
                q_buf["OwnerUserId"].append(_to_int(attrs.get("OwnerUserId")))
                q_buf["Score"].append(_to_int(attrs.get("Score")) or 0)
                q_buf["ViewCount"].append(_to_int(attrs.get("ViewCount")))
                q_buf["AnswerCount"].append(_to_int(attrs.get("AnswerCount")))
                q_buf["Tags"].append(tags)
                q_buf["Title"].append(attrs.get("Title"))
                q_buf["AcceptedAnswerId"].append(
                    _to_int(attrs.get("AcceptedAnswerId")))
                for c in COMMUNITIES:
                    q_buf[f"in_{c}"].append(bool(tag_set & COMMUNITY_SETS[c]))
                q_buf["in_llm"].append(bool(tag_set & LLM_SUBSET))
                n_q += 1
                if len(q_buf["Id"]) >= BATCH_SIZE:
                    flush_q(qw)
            elif pt == "2":
                a_buf["Id"].append(_to_int(attrs.get("Id")))
                a_buf["ParentId"].append(_to_int(attrs.get("ParentId")))
                a_buf["CreationDate"].append(attrs.get("CreationDate"))
                a_buf["OwnerUserId"].append(_to_int(attrs.get("OwnerUserId")))
                a_buf["Score"].append(_to_int(attrs.get("Score")) or 0)
                n_a += 1
                if len(a_buf["Id"]) >= BATCH_SIZE:
                    flush_a(aw)
            else:
                n_other += 1

        p = xml.parsers.expat.ParserCreate()
        p.StartElementHandler = on_start

        CHUNK = 1 << 20
        with POSTS_XML.open("rb") as f:
            while True:
                chunk = f.read(CHUNK)
                if not chunk:
                    p.Parse(b"", True)
                    break
                bytes_read += len(chunk)
                p.Parse(chunk, False)

                now = time.perf_counter()
                if now - last_report > 15:
                    pct = bytes_read / file_size * 100
                    elapsed = now - t0
                    eta = elapsed * (file_size / bytes_read - 1) if bytes_read else 0
                    rps = n_total / elapsed if elapsed else 0
                    print(f"  [{pct:5.1f}%] total={n_total:>11,} "
                          f"Q={n_q:>10,} A={n_a:>10,} | "
                          f"{elapsed/60:5.1f}m | ETA {eta/60:5.1f}m | "
                          f"{rps:>8,.0f} rows/s", flush=True)
                    last_report = now

        flush_q(qw)
        flush_a(aw)
    finally:
        qw.close()
        aw.close()

    elapsed = time.perf_counter() - t0
    q_size = OUT_Q.stat().st_size
    a_size = OUT_A.stat().st_size

    summary = {
        "elapsed_s": elapsed,
        "elapsed_min": elapsed / 60,
        "n_total_rows": n_total,
        "n_questions": n_q,
        "n_answers": n_a,
        "n_other_posttype": n_other,
        "questions_parquet_bytes": q_size,
        "answers_parquet_bytes": a_size,
        "questions_parquet_mb": q_size / 1024**2,
        "answers_parquet_mb": a_size / 1024**2,
        "communities": COMMUNITIES,
        "community_tag_counts": {c: len(s) for c, s in COMMUNITY_SETS.items()},
        "llm_subset_size": len(LLM_SUBSET),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Questions: {n_q:>11,} → {q_size/1024**2:,.1f} MB")
    print(f"  Answers:   {n_a:>11,} → {a_size/1024**2:,.1f} MB")
    print(f"  Other:     {n_other:>11,} (skipped TagWiki/meta)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Posts.xml full-file integrity scan (Phase 1, L2 complete pass).

Design choices:
  - expat SAX parser (xml.parsers.expat) for O(1) memory instead of ElementTree's
    tree-building iterparse — safer for a 96 GB file.
  - Pipe-delimited tag format (|a|b|c|), confirmed from sample pass.
  - Checkpoints every ~1 GB of file read, so an interrupt preserves progress.

Outputs:
  outputs/tables/integrity_full_checkpoint.json  (machine-readable, live-updated)
  outputs/tables/integrity_full_report.md        (human-readable, final)
"""

from __future__ import annotations

import json
import sys
import time
import xml.parsers.expat
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POSTS_XML = ROOT / "Posts.xml"
CHECKPOINT_PATH = ROOT / "outputs" / "tables" / "integrity_full_checkpoint.json"
REPORT_PATH = ROOT / "outputs" / "tables" / "integrity_full_report.md"

# Target tags from CLAUDE_CODE_PROMPT.md — verify each exists and count occurrences.
# NOTE: Stack Overflow uses "reactjs" not "react", "amazon-web-services" not "aws",
# and "vue.js" is the canonical form. We'll still try the prompt's names AND common
# alternates, report both.
TARGET_TAG_ALIASES = {
    "data_science":  ["python", "pandas", "scikit-learn", "numpy"],
    "deep_learning": ["tensorflow", "pytorch", "keras", "neural-network"],
    "llm_ai":        ["chatgpt", "llm", "langchain", "openai-api",
                      "prompt-engineering", "gpt-4", "large-language-model"],
    "web_frontend":  ["javascript", "reactjs", "react", "typescript",
                      "vue.js", "vuejs"],
    "cloud_devops":  ["docker", "kubernetes", "amazon-web-services", "aws",
                      "terraform"],
    "databases":     ["sql", "postgresql", "mongodb", "redis"],
}

# ------------------------------------------------------------------
# Accumulator state (all simple Python types, serializable to JSON)
# ------------------------------------------------------------------

state = {
    "row_count": 0,
    "posttype": Counter(),
    "field_presence": Counter(),
    "missing_owner": 0,
    "first_date": None,
    "last_date": None,
    "yearmonth_rows": Counter(),
    "yearmonth_questions": Counter(),
    "yearmonth_answers": Counter(),
    "tag_counter": Counter(),
    "questions_total": 0,
    "answers_total": 0,
    "questions_with_tags": 0,
    "tag_parse_failures": 0,
}

t_start = time.perf_counter()
file_size = POSTS_XML.stat().st_size
bytes_read_global = 0


def parse_pipe_tags(s: str) -> list[str]:
    """'|wcf|security|spn|' -> ['wcf','security','spn']."""
    return [t for t in s.split("|") if t]


def on_start(name: str, attrs: dict) -> None:
    if name != "row":
        return
    s = state
    s["row_count"] += 1

    for k in attrs:
        s["field_presence"][k] += 1

    pt = attrs.get("PostTypeId", "")
    s["posttype"][pt] += 1

    cd = attrs.get("CreationDate")
    if cd:
        if s["first_date"] is None or cd < s["first_date"]:
            s["first_date"] = cd
        if s["last_date"] is None or cd > s["last_date"]:
            s["last_date"] = cd
        ym = cd[:7]
        s["yearmonth_rows"][ym] += 1
        if pt == "1":
            s["yearmonth_questions"][ym] += 1
        elif pt == "2":
            s["yearmonth_answers"][ym] += 1

    if pt == "1":
        s["questions_total"] += 1
        tags = attrs.get("Tags")
        if tags:
            parsed = parse_pipe_tags(tags)
            if parsed:
                s["questions_with_tags"] += 1
                for t in parsed:
                    s["tag_counter"][t] += 1
            else:
                s["tag_parse_failures"] += 1
    elif pt == "2":
        s["answers_total"] += 1

    if "OwnerUserId" not in attrs:
        s["missing_owner"] += 1


def save_checkpoint(final: bool = False) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "final": final,
        "row_count": state["row_count"],
        "elapsed_s": time.perf_counter() - t_start,
        "bytes_read": bytes_read_global,
        "bytes_total": file_size,
        "progress_pct": (bytes_read_global / file_size * 100) if file_size else 0,
        "first_date": state["first_date"],
        "last_date": state["last_date"],
        "posttype": dict(state["posttype"]),
        "field_presence": dict(state["field_presence"]),
        "missing_owner": state["missing_owner"],
        "questions_total": state["questions_total"],
        "answers_total": state["answers_total"],
        "questions_with_tags": state["questions_with_tags"],
        "tag_parse_failures": state["tag_parse_failures"],
        "yearmonth_rows": dict(state["yearmonth_rows"]),
        "yearmonth_questions": dict(state["yearmonth_questions"]),
        "yearmonth_answers": dict(state["yearmonth_answers"]),
        "n_unique_tags": len(state["tag_counter"]),
        "top_tags": state["tag_counter"].most_common(500),
    }
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_final_report() -> None:
    s = state
    tc = s["tag_counter"]
    total = s["row_count"]
    q = s["questions_total"]

    lines = []
    lines.append("# Posts.xml Integrity Report — Full Scan\n")
    lines.append(f"- **File**: `{POSTS_XML}`")
    lines.append(f"- **Size**: {file_size:,} bytes ({file_size/1024**3:.2f} GB)")
    lines.append(f"- **Total rows**: **{total:,}**")
    lines.append(f"- **Date range**: {s['first_date']} → {s['last_date']}")
    lines.append(f"- **Scan time**: {(time.perf_counter()-t_start)/60:.1f} min "
                 f"({total/(time.perf_counter()-t_start):,.0f} rows/s)")
    lines.append("")

    # PostTypeId
    lines.append("## PostTypeId distribution\n")
    lines.append("| PostTypeId | Label | Count | % |")
    lines.append("|---|---|---:|---:|")
    pt_labels = {"1": "Question", "2": "Answer", "3": "OrphanedTagWiki",
                 "4": "TagWikiExcerpt", "5": "TagWiki",
                 "6": "ModeratorNomination", "7": "WikiPlaceholder",
                 "8": "PrivilegeWiki"}
    for pt, n in sorted(s["posttype"].items(), key=lambda x: -x[1]):
        label = pt_labels.get(pt, "?")
        lines.append(f"| {pt or '(missing)'} | {label} | {n:,} | {n/total*100:.3f}% |")
    lines.append("")

    # Field presence
    lines.append("## Field presence\n")
    lines.append("| Attribute | Rows present | % |")
    lines.append("|---|---:|---:|")
    for k, n in sorted(s["field_presence"].items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {n:,} | {n/total*100:.2f}% |")
    lines.append("")

    # Questions / tags
    lines.append("## Questions & tags\n")
    if q:
        lines.append(f"- Total questions: **{q:,}**")
        lines.append(f"- Questions with parseable Tags: {s['questions_with_tags']:,} "
                     f"({s['questions_with_tags']/q*100:.3f}%)")
        lines.append(f"- Questions where Tags failed to parse: "
                     f"{s['tag_parse_failures']:,}")
    lines.append(f"- Unique tags observed: **{len(tc):,}**")
    lines.append("")

    # Missing owner
    lines.append("## Deleted-user proxy (missing OwnerUserId)\n")
    lines.append(f"- Rows with no OwnerUserId: {s['missing_owner']:,} "
                 f"({s['missing_owner']/total*100:.3f}%)")
    lines.append("")

    # Yearly breakdown
    lines.append("## Yearly volume\n")
    lines.append("| Year | Rows | Questions | Answers |")
    lines.append("|---|---:|---:|---:|")
    by_year_rows = Counter()
    by_year_q = Counter()
    by_year_a = Counter()
    for ym, n in s["yearmonth_rows"].items():
        by_year_rows[ym[:4]] += n
    for ym, n in s["yearmonth_questions"].items():
        by_year_q[ym[:4]] += n
    for ym, n in s["yearmonth_answers"].items():
        by_year_a[ym[:4]] += n
    for y in sorted(by_year_rows):
        lines.append(f"| {y} | {by_year_rows[y]:,} | "
                     f"{by_year_q[y]:,} | {by_year_a[y]:,} |")
    lines.append("")

    # Monthly for DNA window sizing (2015+)
    lines.append("## Monthly question count 2015+ (for DNA window design)\n")
    lines.append("| Year-Month | Questions |")
    lines.append("|---|---:|")
    for ym in sorted(k for k in s["yearmonth_questions"] if k >= "2015"):
        lines.append(f"| {ym} | {s['yearmonth_questions'][ym]:,} |")
    lines.append("")

    # Top tags
    lines.append("## Top 50 tags overall\n")
    lines.append("| Rank | Tag | Count |")
    lines.append("|---:|---|---:|")
    for i, (tag, n) in enumerate(tc.most_common(50), 1):
        lines.append(f"| {i} | `{tag}` | {n:,} |")
    lines.append("")

    # Target tag presence check
    lines.append("## Target-tag coverage check\n")
    lines.append("Source: `CLAUDE_CODE_PROMPT.md` TARGET_TAGS (with common aliases).\n")
    lines.append("| Community | Tag | Count | Exists |")
    lines.append("|---|---|---:|:---:|")
    for community, tags in TARGET_TAG_ALIASES.items():
        for tag in tags:
            n = tc.get(tag, 0)
            exists = "✓" if n > 0 else "✗"
            lines.append(f"| {community} | `{tag}` | {n:,} | {exists} |")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written → {REPORT_PATH}", flush=True)


def main() -> int:
    global bytes_read_global

    if not POSTS_XML.exists():
        print(f"ERROR: {POSTS_XML} not found", file=sys.stderr)
        return 1

    print("=" * 72, flush=True)
    print("Posts.xml Full Scan — expat SAX parser", flush=True)
    print(f"File size: {file_size/1024**3:.2f} GB", flush=True)
    print("=" * 72, flush=True)

    parser = xml.parsers.expat.ParserCreate()
    parser.StartElementHandler = on_start

    CHUNK = 1 << 20            # 1 MB reads
    CHECKPOINT_BYTES = 1 << 30  # 1 GB
    REPORT_SEC = 15

    last_report = t_start
    last_checkpoint = 0

    with POSTS_XML.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                parser.Parse(b"", True)
                break
            bytes_read_global += len(chunk)
            parser.Parse(chunk, False)

            now = time.perf_counter()
            if now - last_report > REPORT_SEC:
                pct = bytes_read_global / file_size * 100
                elapsed = now - t_start
                eta_s = (elapsed * file_size / bytes_read_global) - elapsed \
                    if bytes_read_global else 0
                rps = state["row_count"] / elapsed if elapsed else 0
                print(f"  [{pct:5.1f}%] {state['row_count']:>11,} rows | "
                      f"{elapsed/60:5.1f} min | ETA {eta_s/60:5.1f} min | "
                      f"{rps:>8,.0f} rows/s", flush=True)
                last_report = now

            if bytes_read_global - last_checkpoint > CHECKPOINT_BYTES:
                save_checkpoint(final=False)
                last_checkpoint = bytes_read_global

    save_checkpoint(final=True)
    write_final_report()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted — saving partial checkpoint ...", flush=True)
        save_checkpoint(final=False)
        sys.exit(130)

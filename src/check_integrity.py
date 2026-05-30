"""
Posts.xml integrity check — Phase 1, Layers L1 + sample of L2/L3/L4.

Strategy (per Agent 1 framework):
  L1  file-level:  size, XML header, trailing </posts>
  L2  structural:  row count and PostTypeId distribution (sample)
  L3  schema:      attribute presence rate per row type (sample)
  L4  semantic:    CreationDate range, OwnerUserId NULL rate, Tags format

Scope for this run: first SAMPLE_ROWS rows only (fast feedback before full scan).
"""

from __future__ import annotations

import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POSTS_XML = ROOT / "Posts.xml"

SAMPLE_ROWS = 200_000  # ~a few hundred MB of XML; representative, finishes in minutes
HEAD_BYTES = 2048
TAIL_BYTES = 4096
REPORT_PATH = ROOT / "outputs" / "tables" / "integrity_sample_report.md"


# ---------- L1 ----------

def check_l1(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(HEAD_BYTES)
        f.seek(-TAIL_BYTES, os.SEEK_END)
        tail = f.read()
    return {
        "size_bytes": size,
        "size_gb": size / (1024**3),
        "head_text": head.decode("utf-8", errors="replace"),
        "tail_text": tail.decode("utf-8", errors="replace"),
        "has_xml_header": head.lstrip().startswith(b"<?xml"),
        "has_posts_open": b"<posts" in head,
        "has_posts_close": b"</posts>" in tail,
    }


# ---------- L2 / L3 / L4 on a sample ----------

TAG_PATTERN = re.compile(r"^(<[^<>]+>)+$")  # matches "<tag1><tag2>..."
# Newer dumps sometimes use pipe-delimited tags: "tag1|tag2|tag3" (with a leading/trailing |)
TAG_PATTERN_PIPE = re.compile(r"^\|?[^|]+(\|[^|]+)*\|?$")


def sample_scan(path: Path, max_rows: int) -> dict:
    row_count = 0
    posttype = Counter()
    field_presence = Counter()  # attribute name -> count
    first_date = None
    last_date = None

    questions = 0
    answers = 0
    other = 0

    q_with_tags = 0
    q_tag_format_angle = 0
    q_tag_format_pipe = 0
    q_tag_format_other = 0

    missing_owner = 0  # deleted user rows

    # Track invalid / unexpected rows
    row_type_unknown = 0

    t0 = time.perf_counter()
    last_report_t = t0

    # iterparse streams the file; we only need "end" events for <row>
    # Using huge_tree / no DTD issues since Stack Exchange dumps are self-contained.
    context = ET.iterparse(str(path), events=("end",))

    for _, elem in context:
        if elem.tag != "row":
            elem.clear()
            continue

        row_count += 1
        a = elem.attrib

        for k in a:
            field_presence[k] += 1

        pt = a.get("PostTypeId", "")
        posttype[pt] += 1
        if pt == "1":
            questions += 1
            tags = a.get("Tags")
            if tags:
                q_with_tags += 1
                if TAG_PATTERN.match(tags):
                    q_tag_format_angle += 1
                elif TAG_PATTERN_PIPE.match(tags):
                    q_tag_format_pipe += 1
                else:
                    q_tag_format_other += 1
        elif pt == "2":
            answers += 1
        elif pt == "":
            row_type_unknown += 1
        else:
            other += 1

        cd = a.get("CreationDate")
        if cd:
            if first_date is None or cd < first_date:
                first_date = cd
            if last_date is None or cd > last_date:
                last_date = cd

        if "OwnerUserId" not in a:
            missing_owner += 1

        elem.clear()

        # progress pulse every ~5 s
        now = time.perf_counter()
        if now - last_report_t > 5:
            rate = row_count / (now - t0) if now > t0 else 0
            print(f"  ... {row_count:>8,} rows scanned ({rate:,.0f} rows/s)", flush=True)
            last_report_t = now

        if row_count >= max_rows:
            break

    elapsed = time.perf_counter() - t0

    return {
        "row_count": row_count,
        "elapsed_s": elapsed,
        "rows_per_s": row_count / elapsed if elapsed else 0,
        "posttype_distribution": dict(posttype),
        "questions": questions,
        "answers": answers,
        "other_posttype": other,
        "unknown_posttype": row_type_unknown,
        "field_presence": dict(field_presence),
        "first_date_in_sample": first_date,
        "last_date_in_sample": last_date,
        "questions_with_tags": q_with_tags,
        "tag_format_angle_bracket": q_tag_format_angle,
        "tag_format_pipe": q_tag_format_pipe,
        "tag_format_other": q_tag_format_other,
        "missing_owner_rows": missing_owner,
    }


# ---------- reporting ----------

def write_report(l1: dict, sample: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Posts.xml Integrity Report — Sample Pass\n")
    lines.append(f"- **File**: `{POSTS_XML}`")
    lines.append(f"- **Size**: {l1['size_bytes']:,} bytes ({l1['size_gb']:.2f} GB)")
    lines.append(f"- **XML header present**: {l1['has_xml_header']}")
    lines.append(f"- **`<posts` opening present**: {l1['has_posts_open']}")
    lines.append(f"- **`</posts>` closing present**: {l1['has_posts_close']}")
    lines.append("")
    lines.append("## File head (first 2 KB)\n")
    lines.append("```xml")
    lines.append(l1["head_text"])
    lines.append("```\n")
    lines.append("## File tail (last 4 KB)\n")
    lines.append("```xml")
    lines.append(l1["tail_text"])
    lines.append("```\n")

    lines.append(f"## Sample scan ({sample['row_count']:,} rows)\n")
    lines.append(f"- Scan time: {sample['elapsed_s']:.1f} s "
                 f"({sample['rows_per_s']:,.0f} rows/s)")
    lines.append(f"- Date range in sample: {sample['first_date_in_sample']} "
                 f"→ {sample['last_date_in_sample']}")
    lines.append("")

    lines.append("### PostTypeId distribution\n")
    lines.append("| PostTypeId | Count | % |")
    lines.append("|---|---:|---:|")
    total = sample["row_count"]
    pt_labels = {"1": "Question", "2": "Answer", "3": "OrphanedTagWiki",
                 "4": "TagWikiExcerpt", "5": "TagWiki", "6": "ModeratorNomination",
                 "7": "WikiPlaceholder", "8": "PrivilegeWiki"}
    for pt, n in sorted(sample["posttype_distribution"].items(),
                        key=lambda x: -x[1]):
        label = pt_labels.get(pt, "?")
        lines.append(f"| {pt or '(missing)'} — {label} | {n:,} | {n/total*100:.2f}% |")
    lines.append("")

    lines.append("### Field presence (attributes on `<row>`)\n")
    lines.append("| Attribute | Rows present | % |")
    lines.append("|---|---:|---:|")
    for k, n in sorted(sample["field_presence"].items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {n:,} | {n/total*100:.2f}% |")
    lines.append("")

    q = sample["questions"]
    lines.append("### Question-specific checks\n")
    if q:
        lines.append(f"- Questions in sample: **{q:,}**")
        lines.append(f"- Questions with Tags field: "
                     f"{sample['questions_with_tags']:,} "
                     f"({sample['questions_with_tags']/q*100:.2f}%)")
        lines.append(f"- Tag format `<tag1><tag2>`: "
                     f"{sample['tag_format_angle_bracket']:,}")
        lines.append(f"- Tag format `tag1|tag2|...`: "
                     f"{sample['tag_format_pipe']:,}")
        lines.append(f"- Tag format other / unparseable: "
                     f"{sample['tag_format_other']:,}")
    lines.append("")

    lines.append("### Deleted-user proxy (missing OwnerUserId)\n")
    lines.append(f"- Rows missing OwnerUserId: "
                 f"{sample['missing_owner_rows']:,} "
                 f"({sample['missing_owner_rows']/total*100:.2f}%)")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written → {REPORT_PATH}")


def main() -> int:
    if not POSTS_XML.exists():
        print(f"ERROR: {POSTS_XML} not found", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Posts.xml Integrity Check — Sample Pass")
    print("=" * 70)

    print("\n[L1] File-level checks ...")
    l1 = check_l1(POSTS_XML)
    print(f"  size: {l1['size_gb']:.2f} GB")
    print(f"  has_xml_header     : {l1['has_xml_header']}")
    print(f"  has_posts_open     : {l1['has_posts_open']}")
    print(f"  has_posts_close    : {l1['has_posts_close']}")

    print(f"\n[L2–L4] Sample scan of up to {SAMPLE_ROWS:,} rows ...")
    sample = sample_scan(POSTS_XML, SAMPLE_ROWS)
    print(f"  scanned {sample['row_count']:,} rows in {sample['elapsed_s']:.1f} s")
    print(f"  date range in sample: {sample['first_date_in_sample']} "
          f"→ {sample['last_date_in_sample']}")
    print(f"  PostTypeId: {sample['posttype_distribution']}")

    write_report(l1, sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())

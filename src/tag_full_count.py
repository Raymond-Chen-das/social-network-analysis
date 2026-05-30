"""
Fast pass over Posts.xml to count EVERY tag (not just top 500).

Bypasses the XML parser entirely — scans raw bytes for the pattern
   Tags="|...|"
which (as confirmed by the sample pass) is unambiguous in this dump.

Expected runtime: ~10–15 min on a 96 GB file (I/O + regex only).
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSTS_XML = ROOT / "Posts.xml"
OUT_JSON = ROOT / "outputs" / "tables" / "all_tag_counts.json"

# Pipe-delimited tag values only; the leading/trailing pipe makes this
# pattern extremely unlikely to match anything inside Body or other attributes.
PAT = re.compile(rb'Tags="(\|[^"]+\|)"')
CHUNK = 32 * 1024 * 1024   # 32 MB reads


def main() -> None:
    size = POSTS_XML.stat().st_size
    counter: Counter[str] = Counter()
    questions_seen = 0

    t0 = time.perf_counter()
    last_report = t0
    bytes_read = 0
    carry = b""

    with POSTS_XML.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                # flush any remaining buffer
                for m in PAT.finditer(carry):
                    questions_seen += 1
                    tag_str = m.group(1).decode("utf-8", errors="replace")
                    for t in tag_str.split("|"):
                        if t:
                            counter[t] += 1
                break

            bytes_read += len(chunk)
            buf = carry + chunk

            # Find all complete matches; preserve tail for next iteration
            last_end = 0
            for m in PAT.finditer(buf):
                last_end = m.end()
                questions_seen += 1
                tag_str = m.group(1).decode("utf-8", errors="replace")
                for t in tag_str.split("|"):
                    if t:
                        counter[t] += 1

            # keep the last ~1 KB as carry in case a match straddles the boundary
            carry = buf[max(last_end, len(buf) - 1024):]

            now = time.perf_counter()
            if now - last_report > 15:
                pct = bytes_read / size * 100
                elapsed = now - t0
                eta = elapsed * (size / bytes_read - 1) if bytes_read else 0
                print(f"  [{pct:5.1f}%] questions seen: {questions_seen:>11,} | "
                      f"unique tags: {len(counter):>6,} | "
                      f"elapsed {elapsed/60:5.1f} min | ETA {eta/60:5.1f} min",
                      flush=True)
                last_report = now

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {questions_seen:,} tag strings, "
          f"{len(counter):,} unique tags in {elapsed/60:.1f} min", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({
            "questions_with_tags": questions_seen,
            "unique_tags": len(counter),
            "elapsed_s": elapsed,
            "tag_counts": dict(counter.most_common()),
        }, indent=1),
        encoding="utf-8",
    )
    print(f"Written → {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

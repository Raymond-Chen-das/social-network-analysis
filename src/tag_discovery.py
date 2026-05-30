"""
Tag discovery for TARGET_TAGS curation.

Reads the full-scan checkpoint and groups the top-500 tags into candidate
pools by regex patterns. Output drives the Agent 5 (domain expert)
decision on what each target community should actually contain.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "outputs" / "tables" / "integrity_full_checkpoint.json"
OUT = ROOT / "outputs" / "tables" / "tag_candidates.md"

# Pattern groups — a tag can appear in multiple buckets; human curates.
PATTERNS = {
    "llm_generative_ai": [
        r"\bgpt\b", r"gpt-?\d", r"openai", r"chatgpt", r"chat-?gpt",
        r"\bllm\b", r"large-?language-?model", r"language-?model",
        r"langchain", r"llama", r"claude", r"gemini", r"mistral",
        r"prompt", r"rag\b", r"retrieval-?augmented",
        r"huggingface", r"transformer", r"bert", r"t5\b", r"xlnet",
        r"embedding", r"vector-?db", r"chroma", r"pinecone", r"weaviate",
        r"fine-?tun", r"in-?context",
    ],
    "classical_nlp": [
        r"\bnlp\b", r"natural-?language", r"spacy", r"nltk",
        r"tokeniz", r"word2vec", r"gensim", r"stanza", r"corenlp",
        r"sentiment", r"named-?entity", r"ner\b", r"stemming",
    ],
    "deep_learning": [
        r"tensorflow", r"pytorch", r"\bkeras\b", r"theano", r"caffe",
        r"neural-?network", r"deep-?learning", r"lstm", r"rnn\b",
        r"cnn\b", r"gan\b", r"autoencoder", r"reinforcement-?learning",
        r"conv-?net", r"activation", r"backpropagation",
    ],
    "classical_ml_ds": [
        r"scikit-?learn", r"sklearn", r"\bxgboost\b", r"lightgbm", r"catboost",
        r"\br\b", r"\bnumpy\b", r"\bpandas\b", r"scipy", r"\bmatplotlib\b",
        r"seaborn", r"jupyter", r"\bplotly\b", r"statsmodels",
        r"machine-?learning", r"data-?science", r"classification",
        r"regression", r"clustering", r"random-?forest",
    ],
    "web_frontend": [
        r"^javascript$", r"^reactjs$", r"^react-", r"react-hooks",
        r"^vue\.?js$", r"^angular$", r"angularjs", r"^svelte$",
        r"^typescript$", r"^jsx$", r"^next\.?js$", r"^nuxt",
        r"^html$", r"^css$", r"tailwind", r"^sass$", r"^scss$",
    ],
    "cloud_devops": [
        r"^docker$", r"docker-compose", r"^kubernetes$", r"^k8s$",
        r"^amazon-web-services$", r"^aws-", r"^azure", r"google-cloud",
        r"^gcp$", r"^terraform$", r"ansible", r"jenkins", r"github-actions",
        r"ci-cd", r"helm", r"prometheus", r"grafana",
    ],
    "databases": [
        r"^sql$", r"^postgresql$", r"^mysql$", r"^mongodb$", r"^redis$",
        r"^oracle", r"sqlite", r"^mariadb$", r"^cassandra$",
        r"^elasticsearch$", r"dynamodb", r"^sql-server$", r"nosql",
    ],
}

COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in PATTERNS.items()}


def match_groups(tag: str) -> list[str]:
    return [g for g, regs in COMPILED.items() if any(r.search(tag) for r in regs)]


def main() -> None:
    data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    top = data["top_tags"]  # list of [tag, count]

    # Group tags
    buckets: dict[str, list[tuple[str, int]]] = {g: [] for g in PATTERNS}
    unmatched_high_count = []
    for tag, n in top:
        groups = match_groups(tag)
        if groups:
            for g in groups:
                buckets[g].append((tag, n))
        else:
            # Keep track of top unmatched tags for visibility
            unmatched_high_count.append((tag, n))

    lines = ["# Tag Candidate Pools (from top-500 tags)\n"]
    lines.append(f"Source: `{CHECKPOINT.name}` (top {len(top)} tags)\n")

    for group, tags in buckets.items():
        tags.sort(key=lambda x: -x[1])
        total = sum(n for _, n in tags)
        lines.append(f"## {group}  —  {len(tags)} tags, {total:,} total occurrences\n")
        lines.append("| Tag | Count |")
        lines.append("|---|---:|")
        for t, n in tags:
            lines.append(f"| `{t}` | {n:,} |")
        lines.append("")

    # Also list top 50 unmatched tags to spot anything missed
    unmatched_high_count.sort(key=lambda x: -x[1])
    lines.append("## ⚠️ Top 50 unmatched tags (sanity check — anything we missed?)\n")
    lines.append("| Tag | Count |")
    lines.append("|---|---:|")
    for t, n in unmatched_high_count[:50]:
        lines.append(f"| `{t}` | {n:,} |")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written → {OUT}")


if __name__ == "__main__":
    main()

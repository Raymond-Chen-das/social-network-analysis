"""
Full-coverage tag curation using all 65k+ tag counts.

Produces:
  outputs/tables/tag_pools_full.md     — pattern-grouped tag candidates
  outputs/tables/target_tags_proposed.json  — final TARGET_TAGS recommendation
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAGS_JSON = ROOT / "outputs" / "tables" / "all_tag_counts.json"
OUT_MD = ROOT / "outputs" / "tables" / "tag_pools_full.md"
OUT_JSON = ROOT / "outputs" / "tables" / "target_tags_proposed.json"

# Pattern pools — patterns are *candidates*; we still cut by a minimum count.
POOLS = {
    "llm_generative_ai": [
        r"^gpt", r"-gpt", r"chatgpt", r"^llm", r"-llm",
        r"large-?language-?model", r"^openai", r"langchain",
        r"llama", r"llama-?\d", r"claude(-|$)", r"gemini",
        r"mistral", r"^anthropic", r"cohere",
        r"prompt-engineering", r"prompt-?ing",
        r"^rag$", r"retrieval-aug", r"vector-?database", r"chroma",
        r"^pinecone$", r"weaviate", r"qdrant",
        r"fine-?tuning", r"instruction-?tun",
        r"generative-ai", r"gen-?ai", r"stable-diffusion",
        r"^diffusion", r"midjourney",
    ],
    "nlp_classical_to_transformer": [
        r"^nlp$", r"natural-language", r"^spacy$", r"^nltk$",
        r"transformer", r"^bert", r"-bert", r"^t5$", r"xlnet",
        r"word2vec", r"word-?embedding", r"gensim",
        r"tokeniz", r"huggingface", r"hugging-face",
        r"sentiment-anal", r"named-entity", r"^ner$",
        r"text-classification", r"text-generation",
    ],
    "deep_learning_frameworks_topics": [
        r"^tensorflow", r"^pytorch", r"^keras$", r"^theano$",
        r"^caffe$", r"^mxnet$", r"^onnx$",
        r"neural-network", r"deep-learning",
        r"^lstm$", r"^rnn$", r"^cnn$", r"^gan$", r"autoencoder",
        r"reinforcement-learning", r"convolutional", r"gradient-descent",
        r"backpropagation", r"^activation",
    ],
    "classical_ml_ds": [
        r"scikit-learn", r"^sklearn$", r"^xgboost$", r"^lightgbm$",
        r"^catboost$", r"machine-learning", r"^pandas$", r"^numpy$",
        r"^scipy$", r"^matplotlib$", r"^seaborn$", r"^plotly$",
        r"jupyter", r"statsmodels", r"^r$",
        r"data-science", r"feature-eng",
        r"^classification$", r"^regression$", r"^clustering$",
        r"random-forest", r"decision-tree", r"svm",
    ],
    "web_frontend": [
        r"^javascript$", r"^typescript$", r"^jsx$",
        r"^reactjs$", r"^react-", r"^next\.?js$",
        r"^vue\.?js$", r"^nuxt",
        r"^angular$", r"^angularjs$", r"^angular-",
        r"^svelte$", r"^ember",
        r"^html$", r"^html5$", r"^css$", r"^css3$",
        r"^sass$", r"^scss$", r"tailwind", r"bootstrap-\d",
        r"jquery",
    ],
    "cloud_devops": [
        r"^docker$", r"^docker-", r"^kubernetes$", r"^k8s$",
        r"^helm$", r"^openshift$",
        r"^amazon-web-services$", r"^aws-",
        r"^azure$", r"^azure-",
        r"^google-cloud", r"^gcp$",
        r"^terraform$", r"ansible", r"puppet",
        r"jenkins", r"github-actions", r"gitlab-ci",
        r"ci-cd$", r"^devops$",
        r"prometheus", r"grafana", r"^istio$",
    ],
    "databases": [
        r"^sql$", r"^t-sql$", r"^plsql$", r"^pl-sql$",
        r"^mysql$", r"^postgresql$", r"^postgres$",
        r"^mongodb$", r"^redis$", r"^sqlite$",
        r"^oracle$", r"^oracle\d", r"^mariadb$",
        r"^cassandra$", r"^elasticsearch$",
        r"^dynamodb$", r"^sql-server$", r"^nosql$",
        r"^bigquery$", r"snowflake-cloud",
    ],
    "mobile": [
        r"^android$", r"^ios$", r"^swift$", r"^objective-c$",
        r"^kotlin$", r"^flutter$", r"^dart$",
        r"^react-native$", r"^xamarin$", r"^ionic",
        r"^iphone$", r"^ipad$", r"^xcode$",
        r"android-studio", r"jetpack-compose",
    ],
}

COMPILED = {g: [re.compile(p, re.I) for p in pats]
            for g, pats in POOLS.items()}

# Community assignment priority — in overlap, first match wins.
COMMUNITY_PRIORITY = [
    ("ai_ml",        ["llm_generative_ai", "nlp_classical_to_transformer",
                      "deep_learning_frameworks_topics", "classical_ml_ds"]),
    ("mobile",       ["mobile"]),
    ("web_frontend", ["web_frontend"]),
    ("cloud_devops", ["cloud_devops"]),
    ("databases",    ["databases"]),
]

MIN_COUNT = 100  # exclude tags with < 100 posts (ENA noise floor, per Agent 3)


def match_pools(tag: str) -> list[str]:
    return [g for g, regs in COMPILED.items()
            if any(r.search(tag) for r in regs)]


def main() -> None:
    data = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
    counts = data["tag_counts"]

    # Build pool assignments
    pool_buckets: dict[str, list[tuple[str, int]]] = {g: [] for g in POOLS}
    multi_pool: list[tuple[str, int, list[str]]] = []
    for tag, n in counts.items():
        if n < MIN_COUNT:
            continue
        pools = match_pools(tag)
        if not pools:
            continue
        for p in pools:
            pool_buckets[p].append((tag, n))
        if len(pools) > 1:
            multi_pool.append((tag, n, pools))

    for tags in pool_buckets.values():
        tags.sort(key=lambda x: -x[1])

    # Assign each tag to exactly one community using priority
    community_members: dict[str, list[tuple[str, int, str]]] = {
        name: [] for name, _ in COMMUNITY_PRIORITY
    }
    assigned_tags: set[str] = set()
    for community, pool_names in COMMUNITY_PRIORITY:
        for pool in pool_names:
            for tag, n in pool_buckets[pool]:
                if tag in assigned_tags:
                    continue
                community_members[community].append((tag, n, pool))
                assigned_tags.add(tag)

    # Sort each community by count
    for c in community_members:
        community_members[c].sort(key=lambda x: -x[1])

    # --- write markdown ---
    lines = ["# Tag Pool & Community Curation (full 65k tags)\n"]
    lines.append(f"- Source: `{TAGS_JSON.name}` ({len(counts):,} unique tags)")
    lines.append(f"- Minimum tag count filter: **{MIN_COUNT}**")
    lines.append(f"- Total tags kept: "
                 f"**{sum(1 for n in counts.values() if n >= MIN_COUNT):,}**\n")

    lines.append("## A. Pattern-based pools (raw buckets, tags may appear in multiple)\n")
    for pool, tags in pool_buckets.items():
        total = sum(n for _, n in tags)
        lines.append(f"### {pool} — {len(tags)} tags, {total:,} posts\n")
        lines.append("| Tag | Count |")
        lines.append("|---|---:|")
        for t, n in tags[:50]:
            lines.append(f"| `{t}` | {n:,} |")
        if len(tags) > 50:
            lines.append(f"| ... ({len(tags)-50} more) | |")
        lines.append("")

    lines.append("## B. Proposed 5-community assignment "
                 "(each tag → exactly one community)\n")
    for community, members in community_members.items():
        total = sum(n for _, n, _ in members)
        lines.append(f"### `{community}` — {len(members)} tags, "
                     f"{total:,} posts\n")
        lines.append("| Tag | Count | Sub-pool |")
        lines.append("|---|---:|---|")
        for t, n, p in members[:80]:
            lines.append(f"| `{t}` | {n:,} | {p} |")
        if len(members) > 80:
            lines.append(f"| ... ({len(members)-80} more) | | |")
        lines.append("")

    # Special LLM/GenAI callout for RQ2
    lines.append("## C. 🔍 LLM/GenAI focus table (for RQ2)\n")
    lines.append("Tags that specifically reflect post-2018 LLM/generative-AI "
                 "activity. These drive the pre/post-ChatGPT ENA comparison.\n")
    lines.append("| Tag | Count |")
    lines.append("|---|---:|")
    for t, n in pool_buckets["llm_generative_ai"]:
        lines.append(f"| `{t}` | {n:,} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # --- JSON output for programmatic use ---
    target_tags = {
        community: [t for t, _, _ in members]
        for community, members in community_members.items()
    }
    OUT_JSON.write_text(
        json.dumps({
            "target_tags": target_tags,
            "llm_subset": [t for t, _ in pool_buckets["llm_generative_ai"]],
            "nlp_subset": [t for t, _ in pool_buckets["nlp_classical_to_transformer"]],
            "dl_subset":  [t for t, _ in pool_buckets["deep_learning_frameworks_topics"]],
            "min_count_threshold": MIN_COUNT,
            "totals": {c: sum(n for _, n, _ in m)
                       for c, m in community_members.items()},
        }, indent=2),
        encoding="utf-8",
    )

    print(f"Written → {OUT_MD}")
    print(f"Written → {OUT_JSON}")
    print("\nCommunity sizes (tags, posts):")
    for c, m in community_members.items():
        total = sum(n for _, n, _ in m)
        print(f"  {c:15s}  {len(m):>4} tags   {total:>12,} posts")


if __name__ == "__main__":
    main()

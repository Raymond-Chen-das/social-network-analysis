# Phase 2 network manifest

All graphs live in `data/networks/*.pkl` and can be loaded via `network_builder.load_graph(name)`.

## Tag co-occurrence graphs (undirected)

| name | nodes | edges | density | max w | mean deg | build time |
|---|---:|---:|---:|---:|---:|---:|
| `tag_cooc_global` | 44,324 | 630,505 | 0.0006 | 586,558 | 28.4 | 111.4s |
| `tag_cooc_ai_ml` | 6,879 | 60,462 | 0.0026 | 246,773 | 17.6 | 4.7s |
| `tag_cooc_web_frontend` | 17,121 | 184,111 | 0.0013 | 586,558 | 21.5 | 12.9s |
| `tag_cooc_mobile` | 14,711 | 159,879 | 0.0015 | 281,994 | 21.7 | 8.9s |
| `tag_cooc_cloud_devops` | 8,108 | 70,846 | 0.0022 | 30,628 | 17.5 | 2.8s |
| `tag_cooc_databases` | 9,925 | 91,234 | 0.0019 | 235,817 | 18.4 | 3.6s |
| `tag_cooc_llm_pre_chatgpt` | 371 | 1,494 | 0.0218 | 864 | 8.1 | 1.2s |
| `tag_cooc_llm_post_chatgpt` | 703 | 3,212 | 0.0130 | 773 | 9.1 | 0.2s |

## User Q&A graphs (directed)

| name | nodes | edges | density | max w | mean deg | build time |
|---|---:|---:|---:|---:|---:|---:|
| `user_qa_global` | 5,607,616 | 30,075,497 | 9.56e-07 | 241 | 10.73 | 371.2s |
| `user_qa_ai_ml` | 479,663 | 1,284,368 | 5.58e-06 | 147 | 5.36 | 15.9s |
| `user_qa_web_frontend` | 1,745,947 | 6,187,716 | 2.03e-06 | 112 | 7.09 | 163.5s |
| `user_qa_mobile` | 976,019 | 3,530,799 | 3.71e-06 | 99 | 7.24 | 42.1s |
| `user_qa_cloud_devops` | 490,896 | 829,233 | 3.44e-06 | 59 | 3.38 | 18.9s |
| `user_qa_databases` | 1,068,850 | 2,575,428 | 2.25e-06 | 84 | 4.82 | 32.5s |
| `user_qa_llm_pre_chatgpt` | 3,328 | 2,679 | 2.42e-04 | 4 | 1.61 | 4.9s |
| `user_qa_llm_post_chatgpt` | 6,165 | 4,684 | 1.23e-04 | 4 | 1.52 | 1.7s |
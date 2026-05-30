# Extraction verification

## Counts & coverage

- Questions: **24,101,803** rows (expected 24,101,803)
- Answers:   **35,603,624** rows (expected 35,603,624)
- Questions date range: 2008-07-31T21:42:52.667 → 2024-03-31T23:56:12.250
- Answers   date range: 2008-07-31T22:17:57.883 → 2024-03-31T23:59:19.670
- Null OwnerUserId Q: 506,855 (2.10%)
- Null OwnerUserId A: 421,588 (1.18%)

## Community flag distribution (questions only)

| Community | Questions | % of Q |
|---|---:|---:|
| `ai_ml` | 1,212,671 | 5.03% |
| `web_frontend` | 4,660,782 | 19.34% |
| `mobile` | 2,873,180 | 11.92% |
| `cloud_devops` | 858,785 | 3.56% |
| `databases` | 1,968,874 | 8.17% |
| `llm` (subset of ai_ml) | 10,760 | 0.04% |

## LLM subset pre/post ChatGPT (split 2022-11-30)

- Pre  (< 2022-11-30): **3,599** LLM-tagged questions
- Post (>= 2022-11-30): **7,161** LLM-tagged questions
- Post/Pre ratio: **1.99x**

## Community membership overlap

(how many of the 5 community flags a question has)

| n_communities | questions |
|---:|---:|
| 0 | 13,014,638 |
| 1 | 10,605,562 |
| 2 | 476,087 |
| 3 | 5,508 |
| 4 | 8 |

## Community × year (questions)

| year | total | ai_ml | llm | web_frontend | mobile | cloud_devops | databases |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2008 | 57,376 | 108 | 2 | 4,633 | 968 | 39 | 5,075 |
| 2009 | 340,917 | 1,225 | 1 | 37,419 | 18,882 | 459 | 28,559 |
| 2010 | 689,583 | 4,086 | 6 | 90,152 | 74,322 | 1,356 | 53,355 |
| 2011 | 1,181,811 | 9,522 | 19 | 179,172 | 173,678 | 3,867 | 91,383 |
| 2012 | 1,619,574 | 19,493 | 23 | 266,773 | 238,866 | 9,557 | 129,356 |
| 2013 | 2,028,049 | 36,290 | 29 | 371,604 | 261,429 | 16,043 | 171,843 |
| 2014 | 2,128,799 | 50,754 | 23 | 410,736 | 281,965 | 24,180 | 190,514 |
| 2015 | 2,191,629 | 68,787 | 24 | 429,183 | 299,171 | 37,314 | 190,014 |
| 2016 | 2,197,115 | 89,499 | 33 | 452,515 | 291,552 | 54,882 | 192,054 |
| 2017 | 2,112,176 | 119,221 | 25 | 448,552 | 255,440 | 74,216 | 182,663 |
| 2018 | 1,885,077 | 132,077 | 19 | 383,598 | 210,297 | 92,085 | 155,514 |
| 2019 | 1,762,223 | 141,071 | 143 | 357,708 | 186,546 | 104,781 | 140,287 |
| 2020 | 1,864,330 | 170,626 | 892 | 391,261 | 189,191 | 125,551 | 144,193 |
| 2021 | 1,544,773 | 144,198 | 1,121 | 330,027 | 149,164 | 111,154 | 118,250 |
| 2022 | 1,351,183 | 128,921 | 1,412 | 285,088 | 127,734 | 102,405 | 101,120 |
| 2023 | 954,577 | 81,784 | 5,429 | 186,688 | 94,514 | 83,356 | 63,164 |
| 2024 | 192,611 | 15,009 | 1,559 | 35,673 | 19,461 | 17,540 | 11,530 |

## Tag round-trip sample (first 5 rows)

- Id=38779  Tags=[`wcf`, `security`, `spn`]
  Title: *What SPN do I need to set for a net.tcp service?*
- Id=38784  Tags=[`visual-studio`, `delphi`, `brief-bookmarks`]
  Title: *Visual Studio equivalent to Delphi bookmarks*
- Id=38789  Tags=[`c#`, `asp.net`, `web-services`]
  Title: *Web Service Namespace Dynamic Naming*
- Id=38791  Tags=[`database-design`]
  Title: *Which database table Schema is more efficient?*
- Id=38801  Tags=[`sql`, `sql-server`, `oracle`, `database-design`, `hierarchy`]
  Title: *SQL - How to store and navigate hierarchies?*

Personal research assistant cli 
this is my first projet with langchain to learn more about integration llm and how to made some think usefull
from it by my self 

## How it works

The CLI builds a small LangGraph with three nodes:

```
START -> plan -> search -> research -> END
```

1. **plan** asks the model for a few different search queries (a JSON array).
   If the model returns nothing usable, the raw question is used instead.
2. **search** runs the question plus the planned queries, merges the results
   (duplicate URLs are dropped), ranks them by quality and stores them in the
   graph state (`queries` + `sources` + `context`).
3. **research** sends the question together with the numbered sources to the
   selected model and asks it to cite them as `[1]`, `[2]`, ...

Tune the planner in `.env`: `PLANNER=off` skips the planning call entirely
(one less LLM round trip) and `MAX_QUERIES=3` sets how many extra queries it may
generate.

After the answer, every web source is printed **below it** in a `Sources` table:
number, title, the full clickable URL, and the snippet the search engine
returned. The numbers match the `[1]`, `[2]` citations in the answer, and a
source the model never cited is marked `(not cited in the answer)`. If the answer
contains no citations at all, a warning tells you that every claim is unverified.
The table is built from the search results themselves, so the links stay correct
even when a small local model forgets to add citations.

### Search backends

| Backend | Key needed | Notes |
| --- | --- | --- |
| `duckduckgo` (default) | no | uses the `ddgs` package, with a fallback parser for the DuckDuckGo Lite page |
| `tavily` | `TAVILY_API_KEY` | more reliable, useful when DuckDuckGo rate limits |

If the search fails or returns nothing, the assistant still answers, and prints a
warning saying that the answer is not backed by sources.

### Source quality ranking

Every source the search backends return is scored **0-100** and sorted from the
most to the least reliable before it reaches the model:

| Tier | Score | Examples |
| --- | --- | --- |
| Government | 100 | `nasa.gov`, `noaa.gov`, `gouv.fr`, `europa.eu` |
| University / scientific organization | 85 | `mit.edu`, `cnrs.fr`, `cern.ch`, `ox.ac.uk` |
| Scientific paper | 75 | `arxiv.org`, `pubmed.ncbi.nlm.nih.gov`, `nature.com` |
| Established educational website | 60 | `khanacademy.org`, `britannica.com`, `openstax.org` |
| Wikipedia | 45 | `wikipedia.org`, `wikidata.org` |
| Reddit / forums | 25 | `reddit.com`, `stackoverflow.com`, `youtube.com` |
| Unrecognised domain | 20 | everything else |

The score is used in three places:

1. `search/merger.py` numbers the sources **best first**, so `[1]` in the answer
   points at the strongest source;
2. the research prompt asks the model to build the answer on the highest scored
   sources and to say when a claim only rests on a forum or an unrecognised
   domain;
3. the `Sources` table gains a colour coded `Quality` column (green >= 75,
   yellow >= 45, red below) and the average score appears in its caption.

Domains live in `config.py` (`SOURCE_TIERS` and `SOURCE_DOMAINS`): a listed
domain matches itself and any sub-domain (`fr.wikipedia.org` -> `wikipedia.org`),
and the most specific entry wins, which keeps `pubmed.ncbi.nlm.nih.gov` a paper
instead of a government page. Domains that are not listed fall back on extension
heuristics (`.gov`, `.gouv.fr`, `.edu`, `.ac.uk`, `univ*`, `*forum*`). The
scoring logic itself is in `search/ranking.py`.

Two knobs in `.env`:

```
SOURCE_RANKING=on     # off restores the previous behaviour (scores and column off)
MIN_SOURCE_SCORE=0    # 60 keeps only educational websites and better
```

### Configuration

Copy `.env.example` to `.env` to avoid retyping keys and to change the defaults:

```
OPENAI_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...
SEARCH_PROVIDER=duckduckgo
MAX_SOURCES=5
SOURCE_RANKING=on
MIN_SOURCE_SCORE=0
```

### Run

```
pip install -r requirements.py
python main.py
```

### Tests

```
python -m unittest discover -s tests -v
```

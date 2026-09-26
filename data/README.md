# `data/source_domains.json`

The per-tier domain lists behind the source quality score. Read by
`config.py` at import time and turned into `config.SOURCE_DOMAINS`; the
matching logic is in `search/ranking.py`.

## Format

One key per tier from `SOURCE_TIERS`, each holding a list of bare hostnames.
`unknown` is deliberately absent: an unlisted domain falls through to the
extension heuristics in `ranking.py`, it is never listed.

A listed domain matches itself *and* any sub-domain, and when several entries
match, the most specific one wins. That is what keeps
`pubmed.ncbi.nlm.nih.gov` scored as a paper (75) even though `nih.gov` is
listed as government (100).

## Adding a domain

Bare hostname only. These are all rejected on load, because each one looks
like it works right up until it silently scores 20 as "unrecognised":

| Write this | Not this |
|---|---|
| `nasa.gov` | `https://nasa.gov` |
| `britishcouncil.org` | `www.britishcouncil.org` |
| `learnenglish.britishcouncil.org` | `learnenglish.britishcouncil.org/` |

Listing both `britishcouncil.org` and `learnenglish.britishcouncil.org` is
fine and intentional. Listing the same domain under two different tiers is
an error.

## Why the language-learning domains are there

The eleven entries at the end of `educational` — British Council, Cambridge,
Oxford, YouGlish, VOA, ESL-LAB, English Profile and coe.int — were each
opened by hand before being listed. They are there because without them a
learner question ("give my vocabulary of A1") lands on Quora, blogspam and
unrecognised domains scoring 20-25, instead of on real course material
scoring 60. They were worth a specific fix; please do not prune the list
without checking where the results go first.

## Changing a score

Scores are not here. `SOURCE_TIERS` in `config.py` owns the tier names,
their scores and their labels, and the loader checks the keys here against
it, so renaming a tier fails immediately rather than quietly disabling it.

# Raw Wikipedia articles

One JSON file per breed in `data/breeds_map.json`, downloaded by `scripts/download_wikipedia.py`
through the MediaWiki TextExtracts API (plain text, section headings kept as `== Heading ==`).
Each file records the article `title`, `pageid`, `revid` (the exact revision the text comes from)
and the download time.

These files are committed on purpose: `scripts/ingest.py` builds chunk IDs such as
`kb:siberian-husky:exercise:0` from them, and the golden dataset references those IDs, so every
laptop must ingest exactly the same text. Re-downloading (`--refresh`) can change the IDs.

## Licence and attribution

The text is taken from English Wikipedia and is available under the
[Creative Commons Attribution-ShareAlike 4.0 licence](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0). The authors are the Wikipedia contributors listed in each article's history:
`https://en.wikipedia.org/w/index.php?title=<title>&oldid=<revid>` (use the `title` and `revid` of
the file). Chunks derived from these files are shared under the same licence. Text was converted to
plain text and split into sections; it was not otherwise changed.

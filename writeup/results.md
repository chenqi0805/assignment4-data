# Assignment 4 — computed results (evidence for the writeup)

Every number below was produced by a run on 2026-10-09 in this repo (the
`feat/assignment4-implementation` branch head) on the example Common Crawl data from
`scripts/download_data.py --offline-only` (`local-shared-data/CC/example.warc.gz` +
`example.warc.wet.gz`, first shard of the CC-MAIN-2026-12 crawl). Reproduction commands are
in the last section. This file is working evidence for `writeup.pdf`, not the submission.

## 2.1 `look_at_cc`

**First page of the WARC.** The first record is a `warcinfo` header; the first captured page
is a `request`/`response` pair for
`http://020bld.cn.vauofnj.cn/html/991f998999.html` (`WARC-Date: 2026-03-05T08:39:38Z`).
The page is a Chinese-language content-farm Q&A ("百合花花蕊染色洗得掉吗" — can lily-pistil
stains be washed out) rendered inside an auto-generated corporate template ("昭通市某某橡塑制品维修站",
a rubber-and-plastics repair shop). Accessibility check from this sandbox: the host no longer
resolves (`[Errno -5] No address associated with hostname`), so the page survives only in the
archive — the Internet Archive's role, and Common Crawl's, is exactly this.

**WET quality for this page.** The WET twin holds 1,599 characters against 18,576 bytes of
WARC HTML: markup and scripts are gone, but what remains is mostly template boilerplate
(navigation, "服务热线：020-123456789", placeholder contact blocks) plus visibly mangled spam
text with keyword fragments inserted mid-sentence. The WET format strips the HTML but does
nothing about content quality.

**Where the example is useful / not.** As a corpus for studying boilerplate removal, crawl
provenance, and cleanup tooling it is ideal (real WARC/WET pairs, dead-host pages, spam). As
LM pretraining data it is close to worst-case: template repetition, mangled text, and
multilingual spam dominate the first records, which is precisely what the section-2 filters
have to undo.

**25 annotated WET records.** Language counts over the first 25 conversion records
(lid.176.bin predictions): zh 14, en 4 (one at probability 0.40 on an 8-character "Loading"
placeholder), ru 2, de 1, el 1, ja 1. Page types: content-farm Q&A, gambling spam
(开元棋牌), adult-video spam (records 3, 13, 16, 25), corporate template pages, one Russian
forum, one German forum, a Greek university helpdesk, a UBC alumni directory, and a genuine
company page. None of the first 25 is unambiguously high-quality LM text; the first
plausibly acceptable document is record 8 (UBC alumni directory text, 2,129 characters).
This matches the funnel below: the English filter alone removes ~70% of the file.

## 2.2 `extract_text`

- Fixture check: `tests/test_extract.py::test_extract_text_from_html_bytes` passes —
  resiliparse output matches `moby_extracted.txt` byte-for-byte (detect_encoding → cp1252).
- **2.2b, WARC vs WET on the first page:** extracting the 18,576-byte HTML response with
  `extract_text_from_html_bytes` yields 1,694 characters vs the 1,596-character WET twin.
  The content is the same; the differences are format choices: resiliparse keeps `•` bullets
  for `<li>` items where CC's WET drops them, and BOM/whitespace placement differs. Both
  keep the navigation boilerplate — extraction is not quality filtering.

## 2.3 `language_identification`

Hand audit of 20 random WET documents (seed 336, from the 18,763 documents with ≥ 200
characters): 19/20 predictions correct. The one error: a Spanish-language Peruvian blog
(perutoros.com) predicted `de` at probability 0.31. All five genuinely English documents
scored 0.78–0.96, comfortably above the 0.7 threshold, and the single error would have been
filtered by it anyway. **Recommended threshold: 0.7** — raising it above ~0.78 would start
dropping true English documents, while lowering it below ~0.5 would admit the low-confidence
mispredictions. Funnel effect on the full example WET: 13,793 of 19,633 non-empty records
(70.2%) removed, 5,840 kept.
Downstream risk (writeup 2.3b): language errors mis-label non-English documents as English
(or vice versa); with fastText at threshold 0.7 the wrong-language docs that slip through are
high-confidence ones, and per-record re-checks at every pipeline stage keep the corpus honest.

## 2.4 `mask_pii`

Funnel over the English survivors: **1,402 of 5,840 documents (24.0%) contained at least one
PII match — 2,308 e-mails, 3,442 phone numbers, 119 IPv4 addresses** (a document can contain
several). Formats covered: RFC-style e-mails; US phone formats `(283)-182-3829`,
`(283) 182 3829`, `283-182-3829`, and bare 10-digit numbers; dotted-quad IPv4.

FP/FN observations from the 20-document audit:

- **Partial mask (international format):** the Russian veterinary site evc.ru has
  `+7 (499) 110-3444`; the matcher masks the 10-digit local part and leaves the country code:
  `+7 |||PHONE_NUMBER|||`. A US-format-only phone regex misses country-code prefixes.
- **Upstream redaction:** CC already rewrites some e-mails to `[email protected]` before WET
  export, so nothing is left for our mask — PII coverage depends partly on upstream policy.
- No false positives observed in the sample; every match traced to a real address in the
  document text.

Downstream note (writeup 2.4d): naive PII filtering can destroy legitimate content (support
pages, contact directories); masking (rather than dropping) keeps those documents usable.

## 2.5 `harmful_content`

Dolma/Jigsaw fastText models. Funnel: **NSFW removed 4 documents, toxicity removed 7**
(of the 5,840 English survivors). In the 20-document hand audit: 0/20 NSFW (minimum
probability 0.775 on a benign Finnish newspaper page), 1/20 toxicity false positive — a
Czech hairdressing-education page scored `toxic` at 0.515, i.e. the English-Jigsaw-trained
model fires on non-English text. Note the stage order protects the corpus here: harmfulness
classifiers run *after* the English filter, so non-English pages (where these models are
least reliable) are already gone. The adult-content spam seen in the 2.1 annotation is
overwhelmingly Chinese and never reaches the toxicity stage. For the writeup 2.5c: the
models are sanity filters, not precision instruments — borderline probabilities (0.5–0.6)
should not be trusted as "toxic".

## 2.6 `gopher_quality_filters`

Funnel: **779 of 5,829 documents removed (13.4%), 5,050 kept**. In the 20-document audit,
Gopher passed exactly the 6 content-dense pages (a Russian veterinary hospital, a Canadian
assignment-help site, Kenyan news, TechRadar's archive, an EV-news site, a battery-cabinet
manufacturer) and failed the 14 navigation-heavy template pages — matching my judgment on
all 20. The filters do their job as a coarse boilerplate gate; the earlier fixture tests pin
the exact thresholds (word count 50–100,000, mean word length 3–10, ≤ 30% ellipsis lines,
≥ 80% alphabetic words).

## 2.7 `quality_classifier`

Trained with `scripts/train_quality_classifier.py` (fastText supervised, epochs 15, lr 0.5,
dim 50, word 2-grams, minCount 2, bucket 500,000) on a bounded subsample of the handout's
recipe:

- **Positives:** 12,000 URLs sampled from the external links of one enwiki-20260501
  multistream shard (`enwiki-20260501-pages-articles-multistream1.xml-p1p41242.bz2`),
  scraped and filtered (English, ≥ 300 chars, deduped) → **2,992 documents**
  (7,025 URLs failed to fetch or were non-HTML, 1,100 non-English, 382 duplicates,
  301 without text, 200 too short).
- **Negatives:** **4,333 documents** extracted from the example Common Crawl WARC plus the
  repository's fixture documents, with the same English/length/dedup filters.
- Split 95/5: 6,958 train / 367 validation (wiki: 2,854/138; cc: 4,104/229).

**Results (computed from the saved `quality_classifier.bin`):**
train accuracy **0.9971**, validation accuracy **0.8883**. Fixture sanity checks:
`high_quality_wiki_reference.txt` → `wiki` P = **0.9773**;
`low_quality_cc.txt` → `cc` P = **1.0000**; `tests/test_quality.py::test_classify_quality`
passes. Pipeline effect: removes 4,959 of 5,050 Gopher survivors, keeping 91 documents at
P(wiki) ≥ 0.5. The classifier is not perfect — one kept survivor (cydonix.com, a
machine-translated Russian ebook-spam page) is content-farm material that scored ≥ 0.5 —
so threshold tuning or a larger positive sample would be the next lever. The model binary is
gitignored (`local-shared-data/classifiers/quality_classifier.bin`); the training script is
the reproducible artifact.

## 4 `filter_data` — the funnel (full example WET, single process)

| stage | kept | removed / notes |
|---|---|---|
| input WET conversion records | 19,637 | |
| extract (decode plaintext) | 19,633 | 4 empty |
| english (lid.176.bin P(en) >= 0.7) | 5,840 | 13,793 removed |
| pii masking | 5,840 | 1,402 docs masked (2,308 emails, 3,442 phones, 119 IPs) |
| nsfw (dolma jigsaw) | 5,836 | 4 removed |
| toxic (dolma jigsaw) | 5,829 | 7 removed |
| gopher rules (2.6) | 5,050 | 779 removed |
| quality (P(wiki) >= 0.5) | 91 | 4,959 removed |
| exact document dedup | 91 | 0 removed |
| minhash lsh dedup (final kept) | 91 | 0 removed |
| record errors (dropped) | 0 | |
| characters in -> out | 2,224,747 | 135,012,292 in (1.6%) |

Every row sums exactly (each record ends in exactly one terminal event — removed at a stage,
kept, or errored; zero errors on this file). **Runtime: 35.7 s** for the whole file in one
process, including model loading (the three fastText models dominate RAM at ~2.1 GB, not
time). Projection: 2,500 WETs ≈ 24.8 CPU-hours sequential; with the Modal fan-out
(128 containers, handout 4.2) that is ~12 minutes of wall-clock per 2,500 files, and a full
Common Crawl dump scales linearly in file count. The language filter dominates removals
(70%), the quality classifier dominates the rest (98% of Gopher survivors).

**Dedup observations.** Within this file's 91 survivors there were no exact or near
duplicates (both dedup stages removed 0) — the file is a crawl shard, so cross-file dedup at
corpus scale is where duplicate removal happens. The dedup implementations themselves are
pinned by `tests/test_deduplication.py`: exact line dedup drops *every* occurrence of a
repeated line (including emptied documents), and MinHash/LSH keeps exactly one document per
near-duplicate cluster — on the fuzzy-license fixtures the rails/react MIT pair has word
5-gram Jaccard 0.9191 and collapses to one survivor at threshold 0.8.

## 4 `inspect_filtered_data`

`data/smoke/inspect_report.md` (generated by `scripts/inspect_filtered_data.py --stats-dir`)
contains the funnel table above plus reservoir-sampled evidence (seed 336): kept documents
for quality judgment, per-filter removal examples, and before/after pairs for masked
documents. Spot judgments of kept samples: sample 1 (chestofbooks.com encyclopedia entry,
39k chars) and sample 2 (aztecglyphs.wired-humanities.org, 2.2k chars) are genuinely
reference-grade text; sample 3 (cydonix.com, 48k chars) is machine-translated ebook spam
that the quality classifier wrongly kept — the honest failure case to quote in the writeup.

## 4 `tokenize_data`

GPT-2 tokenizer, `<|endoftext|>` appended after each document, `np.uint16` serialization.
Result: **546,880 tokens** for the 91 kept documents (2,224,747 characters → 4.07
characters/token), written to `data/smoke/train.bin` (exactly 546,880 × 2 = 1,093,760 bytes).
**Paloma contamination guard (handout 5.4): all 91 documents checked against the decoded
Paloma C4-100 validation set — 0 overlaps**; any overlap is a hard error that blocks the
output file.

## 4 `train_model` — not run here (no GPU / no Modal credentials in this workspace)

The wiring is in place; a user with Modal credentials runs:

```bash
# one-time: set SUNET_ID in cs336_data/modal_utils.py (currently a TODO), then:
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin     # 8x B200, ~2 h
uv run modal run scripts/generate_with_gpt2_tok.py --model-path /root/data/output/your_data
```

Config lives in `cs336_basics/train_config.py` (GPT-2-small-shaped ~430M params, 16,384
steps, cosine 6e-4 → 6e-5, Paloma C4-100 eval every 2,000 steps). `scripts/train.py` must
not be modified.

## Reproduction commands

```bash
uv run python scripts/download_data.py --offline-only          # models + example WARC/WET (~2.1 GB)
uv run pytest ./tests -q                                       # 21 passed
uv run python scripts/filter_data.py \
  --input local-shared-data/CC/example.warc.wet.gz \
  --output-dir data/smoke/filtered --samples 5                 # funnel table + stats sidecar
uv run python scripts/inspect_filtered_data.py \
  --stats-dir data/smoke/filtered --output-report data/smoke/inspect_report.md
uv run python scripts/tokenize_data.py \
  --input data/smoke/filtered --output data/smoke/train.bin    # contamination check + uint16 counts
uv run python scripts/train_quality_classifier.py urls|scrape|negatives|train
```

The local paths of all three section-4 scripts run with no Modal import (the sandbox has no
Modal credentials; `SUNET_ID` in `cs336_data/modal_utils.py` remains a TODO for Qi).

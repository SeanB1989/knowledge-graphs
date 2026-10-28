"""
Calliope Corpus Retrieval (v2)
==============================

This is the file that answers "how does it actually find things?"

THE SHORT ANSWER TO YOUR QUESTION
---------------------------------
v1 did exactly what you suspected: a single nearest-neighbour lookup in
vector space, top-5, no filtering, no ranking beyond raw cosine distance.
"Rough and ready semantic globs" is a fair description. It was one blunt
instrument doing a job that wants three.

v2 runs a three-stage pipeline. Each stage exists because the previous one
has a specific, known failure mode.

  STAGE 1 -- RECALL (cast a wide net, ~40 candidates)
  ---------------------------------------------------
  Two independent searches run, because they fail differently:

  (a) DENSE / vector search. Embed the query, find chunks whose vectors are
      closest. Good at meaning: "alienation under late capitalism" will find a
      passage about "the worker estranged from his product" with no shared
      words. Bad at specifics: it will happily return something thematically
      adjacent when you asked for an exact name, term of art, or quotation,
      because rare proper nouns barely move a 768-dimensional average.

  (b) SPARSE / BM25 keyword search. The classic search-engine algorithm. It
      scores a chunk by how often the query's words appear in it, discounted
      by how common those words are across the whole corpus (so "the" counts
      for nothing and "Ngundeng" counts for a lot), and normalised by chunk
      length so long chunks don't win by default. Good at exactly what dense
      search is bad at: names, jargon, quotations, rare terms. Bad at
      synonyms -- it cannot know "estranged" relates to "alienation".

  Their errors are close to uncorrelated, which is the whole point. This is
  called hybrid retrieval and it is the single biggest cheap win available.

  The two ranked lists are merged with Reciprocal Rank Fusion: each chunk
  scores sum(1 / (60 + rank)) across the lists it appears in. RRF uses only
  RANK, never the raw scores, which is what makes it safe -- cosine
  similarities and BM25 scores live on incomparable scales and averaging them
  directly is meaningless.

  STAGE 2 -- PRECISION (rerank the candidates down to the best few)
  -----------------------------------------------------------------
  A cross-encoder reads the query and each candidate chunk TOGETHER, as one
  input, and scores the pair. This is fundamentally more accurate than stage 1:
  a bi-encoder has to compress a whole chunk into one vector before it has any
  idea what will be asked, whereas a cross-encoder gets to look at both sides
  at once and attend across them. It is far too slow to run over 1,000 chunks,
  which is exactly why it goes second, over a shortlist.

  In practice this is where "roughly on topic" becomes "actually answers it."

  STAGE 3 -- HONESTY (decide what is worth showing at all)
  --------------------------------------------------------
  Three things v1 never did:
    * A relevance FLOOR. v1 always returned 5 results, so a query with nothing
      relevant in the corpus still got 5 confident-looking passages back, and
      Calliope would dutifully cite them. Now weak matches are labelled or
      withheld, and "nothing in the corpus speaks to this" is a real outcome.
    * SOURCE DIVERSITY. One book can no longer monopolise every slot. The Nuer
      is half the corpus by chunk count; without a cap it dominates queries it
      has nothing to say about.
    * CONFIDENCE LABELS. Every result carries strong/moderate/weak, so
      Calliope's prompt can tell her to lean on strong hits and hedge on weak
      ones rather than treating all retrieved text as equally authoritative.

WHAT ABOUT TAGS AND A DIFFERENT DATABASE?
-----------------------------------------
You asked. Honest answers:

  TAGS -- worth it, but not as a replacement for any of the above. Tags help
  with FILTERING (only Sean's writing / only Fisher / only chapter 3), not
  with ranking. They're supported here via the `where` filters below, and the
  ingest now records provenance, author, title and page automatically. Manual
  topic tags on ~1000 chunks would be a lot of labour for a small gain; if you
  want them, generate them with the local model at ingest time rather than by
  hand.

  A DIFFERENT DB -- no. Chroma is not the bottleneck and won't be until you're
  past roughly a million chunks. You're at ~1,000. Swapping in Qdrant or
  pgvector would change nothing about answer quality. The bottleneck was, in
  order: garbage input text, a truncating embedding model, and no reranking.
  Storage was never on the list.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(HERE, "data", "chroma")
COLLECTION_NAME = (
    f"calliope_corpus_{embeddings.EMBED_SPACE_VERSION}".replace(".", "_").replace("-", "_")
)

# Stage-1 breadth. Larger = better recall, slightly slower stage 2.
CANDIDATES_PER_METHOD = 25
RRF_K = 60          # standard RRF damping constant
RERANK_POOL = 30    # how many fused candidates the cross-encoder scores

# -- Relevance thresholds -----------------------------------------------------
# These are MEASURED, not guessed. Run:  python3 corpus.py --calibrate
#
# The starting values below are a generic guess at the cross-encoder's range,
# and a generic guess is exactly what keeps going wrong here. ms-marco
# cross-encoders are trained on short web-search queries against factual
# passages; against abstract prose their whole score range shifts down, so
# thresholds that look sane in the abstract reject obviously-correct matches.
# Observed on this corpus before calibration: the essay "Nameless Dread"
# scored -2.71 for the query "what have I written about dread" -- an exact
# title match, rated below the cutoff and discarded.
#
# --calibrate probes the corpus with nonsense and genuine queries, measures
# where each actually lands, and writes data/calibration.json. That file wins
# over everything here.
STRONG = 2.0
MODERATE = -2.0
FLOOR = -6.0        # below this, the passage is not shown at all

CALIBRATION_PATH = os.path.join(HERE, "data", "calibration.json")


def _load_calibration() -> dict | None:
    """Apply measured thresholds from disk, if --calibrate has been run."""
    global STRONG, MODERATE, FLOOR, ABS_BASELINE
    try:
        with open(CALIBRATION_PATH) as f:
            cal = json.load(f)
    except (OSError, ValueError):
        return None
    STRONG = float(cal.get("strong", STRONG))
    MODERATE = float(cal.get("moderate", MODERATE))
    FLOOR = float(cal.get("floor", FLOOR))
    if "abs_baseline" in cal and "CALLIOPE_ABS_BASELINE" not in os.environ:
        ABS_BASELINE = float(cal["abs_baseline"])
    return cal

MAX_PER_SOURCE = 2  # diversity cap, relaxed if too few results survive

# Calibration for the no-reranker fallback.
#
# ABS_BASELINE is where "unrelated" actually sits -- measured, not guessed.
# Measured on the real corpus (602 chunks, Nomic v1.5) via --calibrate:
# nonsense queries topped out at 0.547, genuine queries bottomed out at 0.731.
# 0.64 sits between them, close enough to the noise ceiling that nonsense
# lands below the "moderate" line rather than squeaking over it.
#
# This IS corpus- and model-specific. Re-measure after any embedding change or
# substantial corpus change:  python3 corpus.py --calibrate
ABS_BASELINE = float(os.environ.get("CALLIOPE_ABS_BASELINE", "0.64"))
ABS_SCALE = 25.0

RERANK_MODEL = os.environ.get(
    "CALLIOPE_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

_collection = None
_docs: list[str] = []
_metas: list[dict] = []
_ids: list[str] = []
_vecs = None          # numpy array of unit-normalised chunk embeddings
_bm25 = None
_reranker = None
_rerank_unavailable_reason = ""


# -- BM25 ---------------------------------------------------------------------
# Implemented inline rather than pulled from a package: it is ~40 lines, the
# corpus is small enough that a pure-Python version is instant, and it means
# one less dependency that can break an offline setup.

_TOKEN = re.compile(r"[a-z0-9']+")

# Words too common to carry signal. Deliberately short -- BM25's IDF term
# already down-weights common words; this just saves a little work.
_STOP = frozenset("""
a an and are as at be been but by for from had has have he her his i if in is it
its of on or she that the their them then there these they this to was were
what when which who will with would you your
""".split())


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25:
    """
    Okapi BM25. For each query term t and document d:

        score(d, t) = IDF(t) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * |d|/avgdl))

    where f is the term's frequency in d, |d| is the doc length, avgdl the
    average doc length, and IDF(t) rewards terms that are rare corpus-wide.

    k1 controls how quickly repeated occurrences stop helping (saturation);
    b controls how strongly long documents are penalised.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        # Inverted index: term -> list of (doc_index, term_frequency).
        # Scoring then touches only the documents that actually contain a query
        # term, instead of sweeping all N documents per term.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(corpus_tokens):
            for term, f in Counter(toks).items():
                self.postings[term].append((i, f))

        # Robertson/Sparck-Jones IDF with +0.5 smoothing, floored at a small
        # positive so ubiquitous terms contribute ~nothing rather than going
        # negative and actively penalising a match.
        self.idf = {
            t: max(1e-6, math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)))
            for t, p in self.postings.items()
        }

    def scores(self, query: str) -> list[float]:
        out = [0.0] * self.N
        if not self.N or not self.avgdl:
            return out
        for t in _tokenize(query):
            postings = self.postings.get(t)
            if not postings:
                continue
            idf = self.idf[t]
            for i, f in postings:
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


# -- Loading ------------------------------------------------------------------

def _load():
    """Load the collection and build the in-memory BM25 index. Idempotent."""
    global _collection, _docs, _metas, _ids, _bm25, _vecs
    if _collection is not None:
        return

    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(f"Corpus dependencies missing: {e}\nRun: pip3 install chromadb") from e

    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError("Corpus index not found. Run: python3 ingest.py")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        available = [c.name for c in client.list_collections()]
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found (available: {available}).\n"
            f"The embedding model changed, so the old index is unusable. "
            f"Run: python3 ingest.py --rebuild"
        ) from e

    import numpy as np

    data = collection.get(include=["documents", "metadatas", "embeddings"])
    docs = data["documents"] or []
    if not docs:
        raise RuntimeError("Corpus index is empty. Run: python3 ingest.py --rebuild")

    vecs = np.asarray(data["embeddings"], dtype="float32")

    # Guard against bad vectors before they poison everything downstream.
    #
    # np.maximum(nan, 1e-12) returns NAN -- numpy propagates it rather than
    # clamping -- so a single chunk whose embedding came back with a NaN or inf
    # survives normalisation and then contaminates the whole similarity matrix
    # on every query. That surfaces as "divide by zero / overflow / invalid
    # value encountered in matmul", which points at the matmul but is really a
    # load-time problem.
    #
    # Zeroing a bad row means it scores 0.0 against every query and is simply
    # never retrieved, which is the correct outcome for a chunk we cannot
    # represent.
    bad = ~np.isfinite(vecs).all(axis=1)
    if bad.any():
        vecs[bad] = 0.0
        print(f"  [corpus] {int(bad.sum())} chunk(s) had unusable embeddings "
              f"and were excluded. Re-run: python3 ingest.py --rebuild",
              file=sys.stderr)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms == 0)] = 1.0
    vecs /= norms

    _ids = data["ids"]
    _docs = docs
    _metas = data["metadatas"] or [{} for _ in docs]
    _vecs = vecs
    _bm25 = BM25([_tokenize(d) for d in _docs])
    _load_calibration()
    _collection = collection   # set last: a failure above leaves state unloaded


def _get_reranker():
    """Load the cross-encoder lazily. Returns None if unavailable."""
    global _reranker, _rerank_unavailable_reason
    if _reranker is not None or _rerank_unavailable_reason:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    except Exception as e:
        # Not fatal. Without it we fall back to fused rank order, which is
        # still far better than v1 -- just less precise.
        _rerank_unavailable_reason = f"{type(e).__name__}: {e}"
        return None
    return _reranker


# -- Retrieval ----------------------------------------------------------------

def _dense_ranking(query: str, where: dict | None):
    """
    Return (ranked indices, {index: similarity}, distribution stats).

    Computes exact cosine similarity against every chunk rather than asking
    Chroma for approximate nearest neighbours. Two reasons:

      * At ~1,100 chunks this is a single matrix multiply -- roughly a
        millisecond. Approximate search buys nothing at this scale.
      * It gives us the FULL similarity distribution for this query, not just
        the top 25. That distribution is what makes the calibration below
        possible, and it is not recoverable from a top-k result set.
    """
    import numpy as np

    vec = np.asarray(embeddings.embed_one(query, kind="query"), dtype="float32")
    if not np.isfinite(vec).all():
        raise RuntimeError(
            "The embedding model returned a non-finite vector for this query. "
            "Reload the embedding model in LM Studio and try again."
        )
    norm = float(np.linalg.norm(vec))
    vec = vec / norm if norm > 0 else vec

    # numpy on macOS (Accelerate/vecLib BLAS) raises spurious "divide by zero /
    # overflow / invalid value encountered in matmul" warnings on perfectly
    # finite float32 input. The load-time guard above already removes any
    # genuinely bad vector, so we suppress here and then VERIFY -- if the
    # result really is non-finite, that is a bug worth surfacing, not hiding.
    with np.errstate(all="ignore"):
        sims_all = _vecs @ vec      # _vecs is pre-normalised, so this is cosine
    if not np.isfinite(sims_all).all():
        sims_all = np.nan_to_num(sims_all, nan=-1.0, posinf=-1.0, neginf=-1.0)
        print("  [corpus] some similarities were non-finite and were discarded; "
              "re-run: python3 ingest.py --rebuild", file=sys.stderr)

    idx = np.arange(len(_docs))
    if where:
        mask = np.array([_matches_where(m, where) for m in _metas])
        idx = idx[mask]
        if idx.size == 0:
            return [], {}, {"median": 0.0, "p90": 0.0, "spread": 1.0}

    subset = sims_all[idx]

    # Baseline for THIS query against THIS corpus. Most chunks are irrelevant
    # to any given question, so the median is a direct measurement of "what
    # does unrelated look like right now" -- no constant required.
    median = float(np.median(subset))
    p90 = float(np.percentile(subset, 90))
    spread = max(p90 - median, 1e-6)

    top = idx[np.argsort(-subset)[:min(CANDIDATES_PER_METHOD, idx.size)]]
    order = [int(i) for i in top]
    sims = {int(i): float(sims_all[i]) for i in top}
    return order, sims, {"median": median, "p90": p90, "spread": spread}


def _calibrated_score(sim: float, stats: dict) -> float:
    """
    Convert a raw cosine similarity into the same scale the cross-encoder
    thresholds use, calibrated against this query's own distribution.

    A MEASURED THRESHOLD, NOT AN INTUITED ONE
    -----------------------------------------
    The first version of this used 0.45 as the relevance boundary, picked by
    intuition. That was wrong: on this corpus a genuinely unrelated pair scores
    0.542, comfortably above it. Nomic's similarities sit in a narrow band high
    up the scale and almost nothing approaches zero, so a guessed cutoff waves
    unrelated text through as "moderate" -- the exact failure the relevance
    floor exists to prevent. ABS_BASELINE is now measured with --calibrate.

    WHY NOT SCORE PURELY RELATIVE TO THE QUERY'S OWN DISTRIBUTION
    -------------------------------------------------------------
    A tempting alternative is to ignore absolute values and ask only "how far
    above the median does this sit", in units of (p90 - median). It is
    appealingly self-calibrating. Measured against this corpus, it is also
    backwards:

        query                     top    median   p90    z
        [noise] carburettor      0.547   0.443   0.482  2.67
        [real]  the historian    0.818   0.632   0.723  2.04

    The nonsense query scores HIGHER. The reason is structural: when a query
    genuinely matches the corpus, many chunks are somewhat relevant, so the
    whole distribution shifts up and widens, and the best hit no longer stands
    out from its neighbours. When a query matches nothing, the distribution is
    tight and any small lead looks dramatic. Relative scoring rewards exactly
    the wrong thing.

    So the absolute measure decides, and the relative one is kept only as a
    floor: a chunk at or below the median is no better than a random chunk, and
    is rejected outright regardless of what its raw similarity looks like.
    """
    z = (sim - stats["median"]) / stats["spread"]

    # A chunk at or below the median is, by definition, no more similar to the
    # query than a randomly chosen one. Nothing else matters; reject it.
    if z <= 0:
        return FLOOR - 1.0

    return (sim - ABS_BASELINE) * ABS_SCALE


def _sparse_ranking(query: str, allowed: set[int] | None) -> list[int]:
    """Return candidate indices ranked by BM25 keyword score."""
    scores = _bm25.scores(query)
    idx = [i for i, s in enumerate(scores) if s > 0]
    if allowed is not None:
        idx = [i for i in idx if i in allowed]
    idx.sort(key=lambda i: scores[i], reverse=True)
    return idx[:CANDIDATES_PER_METHOD]


def _matches_where(meta: dict, where: dict) -> bool:
    return all(meta.get(k) == v for k, v in where.items())


def search_corpus(query: str, n_results: int = 5, where: dict | None = None,
                  include_weak: bool = False, strict_diversity: bool = False) -> dict:
    """
    Run the full three-stage retrieval pipeline.

    Returns a dict with `results` (list of passages) plus `diagnostics`, so
    the caller can see WHY it got what it got.

    where: optional metadata filter, e.g. {"provenance": "sean"} or
           {"author": "Mark Fisher"}.
    strict_diversity: keep the per-source cap even when this returns fewer hits.
    """
    _load()
    n_results = max(1, min(int(n_results), 10))

    # --- Stage 1: recall -----------------------------------------------------
    dense_order, sims, dist = _dense_ranking(query, where)

    allowed = None
    if where:
        allowed = {i for i, m in enumerate(_metas) if _matches_where(m, where)}
    sparse_order = _sparse_ranking(query, allowed)

    dense_set, sparse_set = set(dense_order), set(sparse_order)

    fused: dict[int, float] = defaultdict(float)
    for rank, i in enumerate(dense_order):
        fused[i] += 1.0 / (RRF_K + rank + 1)
    for rank, i in enumerate(sparse_order):
        fused[i] += 1.0 / (RRF_K + rank + 1)

    candidates = sorted(fused, key=lambda i: fused[i], reverse=True)[:RERANK_POOL]
    if not candidates:
        return {
            "query": query,
            "results": [],
            "diagnostics": {
                "verdict": "nothing in the corpus is relevant to this query",
                "dense_hits": 0, "sparse_hits": 0, "fused_candidates": 0,
                "corpus_chunks": len(_docs), "scoring": "n/a",
                "filter": where or None, "reranker_note": None,
                "strict_diversity": strict_diversity,
            },
        }

    # --- Stage 2: precision --------------------------------------------------
    reranker = _get_reranker()
    if reranker is not None:
        pairs = [(query, _docs[i]) for i in candidates]
        rr = reranker.predict(pairs)
        scored = sorted(zip(candidates, (float(x) for x in rr)),
                        key=lambda t: t[1], reverse=True)
        scoring = "cross-encoder"
    else:
        # No reranker. The naive fallback -- give every candidate the same
        # neutral pseudo-score -- would quietly disable stage 3 entirely:
        # everything would clear the floor, so "nothing in the corpus is
        # relevant" could never fire, which is the single behaviour this
        # rewrite most needed to add. Losing precision when a model is missing
        # is acceptable; losing honesty is not.
        #
        # So we derive a score from cosine similarity instead, mapped onto the
        # same scale the thresholds use. For Nomic embeddings, ~0.45 is the
        # rough boundary between "topically related" and "unrelated", so that
        # maps to 0 and each 0.05 of similarity is worth one point.
        scored = []
        for i in candidates:
            if i in sims:
                scored.append((i, _calibrated_score(sims[i], dist)))
            else:
                # Keyword-only hit: BM25 matched a rare term, which is real
                # evidence but not on this scale. Park it just above the floor
                # so it survives but never reads "strong".
                scored.append((i, MODERATE + 0.5))
        scored.sort(key=lambda t: t[1], reverse=True)
        scoring = (f"calibrated-cosine (reranker unavailable; "
                   f"baseline {dist['median']:.3f}, spread {dist['spread']:.3f})")

    # --- Stage 3: honesty ----------------------------------------------------
    kept: list[tuple[int, float]] = []
    per_source: Counter = Counter()
    floor = FLOOR if include_weak else MODERATE

    for i, score in scored:
        if len(kept) >= n_results:
            break
        if score < floor:
            continue
        src = _metas[i].get("source", "?")
        if per_source[src] > MAX_PER_SOURCE:
            continue
        per_source[src] += 1
        kept.append((i, score))

    # If the diversity cap starved the result set, relax it -- but never drop
    # below the hard FLOOR, which is what makes "nothing relevant" possible.
    if not strict_diversity and len(kept) < min(2, n_results):
        chosen = {i for i, _ in kept}
        for i, score in scored:
            if len(kept) >= n_results:
                break
            if i in chosen or score < FLOOR:
                continue
            kept.append((i, score))
            chosen.add(i)

    results = []
    for i, score in kept:
        m = _metas[i]
        p0, p1 = m.get("page_start"), m.get("page_end")
        if p0 and p1:
            loc = f"p. {p0}" if p0 == p1 else f"pp. {p0}-{p1}"
        else:
            loc = None
        author = m.get("author", "Unknown")
        title = m.get("title", "Unknown")
        results.append({
            # The stable id this chunk carries in Chroma: md5(stem|index|text[:200]),
            # minted in ingest.py. It was loaded into _ids all along and never
            # surfaced, which meant Calliope could read a passage but had no way to
            # REFER to it -- so no graph assertion could ever point at its evidence.
            # This is the join key between the corpus and the ontology.
            "chunk_id": _ids[i] if i < len(_ids) else None,
            "text": _docs[i],
            "author": author,
            "title": title,
            "source": m.get("source"),   # file stem; the Text-node key
            "location": loc,
            "provenance": m.get("provenance", "published"),
            "citation": f"{author}, {title}" + (f", {loc}" if loc else ""),
            "relevance": round(score, 2),
            "confidence": ("strong" if score >= STRONG
                           else "moderate" if score >= MODERATE
                           else "weak"),
            "similarity": round(sims[i], 3) if i in sims else None,
            "found_by": ("both" if i in dense_set and i in sparse_set
                         else "meaning" if i in dense_set
                         else "keyword"),
        })

    strong = sum(1 for r in results if r["confidence"] == "strong")
    if not results:
        verdict = "nothing in the corpus is relevant to this query"
    elif strong:
        verdict = f"{strong} strong match(es)"
    else:
        verdict = "only loose matches -- treat as suggestive, not authoritative"

    return {
        "query": query,
        "results": results,
        "diagnostics": {
            "verdict": verdict,
            "scoring": scoring,
            "corpus_chunks": len(_docs),
            "dense_hits": len(dense_order),
            "sparse_hits": len(sparse_order),
            "fused_candidates": len(candidates),
            "filter": where or None,
            "strict_diversity": strict_diversity,
            "similarity_baseline": round(dist["median"], 3),
            "similarity_p90": round(dist["p90"], 3),
            "reranker_note": _rerank_unavailable_reason or None,
        },
    }


# The smallest passage excerpt still worth showing. Below this a quote carries
# no argument, only a gesture at one, so it is better to drop the result whole
# and say we dropped it.
MIN_EXCERPT_CHARS = 320

# Must stay at or below TOOL_RESULT_CHAR_CAP in config.py. Kept slightly under
# so main.py's hard cap never fires on a corpus result.
DEFAULT_CHAR_BUDGET = 3800


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " [...]"


def format_for_tool(query: str, n_results: int = 5, where: dict | None = None,
                    char_budget: int = DEFAULT_CHAR_BUDGET,
                    strict_diversity: bool = False) -> str:
    """
    JSON string for use as a tool result in Calliope's loop, guaranteed to fit
    in char_budget and guaranteed to still be valid JSON.

    WHY THIS IS BUDGET-AWARE (measured 2026-08-13)
    ----------------------------------------------
    v1 trimmed each passage to 1200 chars and returned whatever that came to.
    Measured against the real 2,780-chunk corpus, a 5-result payload came to
    6,334 characters. main.py then applied TOOL_RESULT_CHAR_CAP = 4000 with a
    blind slice:

        result = result[:TOOL_RESULT_CHAR_CAP] + "\n... [truncated]"

    That discarded 37% of the payload, and because each result serialises text
    FIRST and its metadata after, the cut landed inside result 3's passage. Of
    five results the model received author and citation for TWO. The remaining
    three arrived as anonymous prose with no attribution, and the whole string
    was no longer parseable JSON.

    This is why assert_passage never fired once in the graph's lifetime: that
    tool REQUIRES author and title, and for most results those fields were
    being deleted in transit. The model was not being stubborn. It was being
    handed passages it had no lawful way to cite.

    So: shrink excerpts to fit, never drop metadata, drop whole results from the
    weak end if it still will not fit, and SAY how many were dropped rather than
    letting the model believe it saw everything.
    """
    try:
        payload = search_corpus(query, n_results=n_results, where=where)
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    full = payload["results"]

    def render(results, omitted):
        p = dict(payload)
        p["results"] = results
        diag = dict(p["diagnostics"])
        if omitted:
            diag["results_omitted_for_budget"] = omitted
            diag["omission_note"] = (
                f"{omitted} lower-ranked result(s) were dropped to fit the context "
                f"budget. Do not treat this list as the complete set of matches."
            )
        p["diagnostics"] = diag
        return json.dumps(p, indent=2, ensure_ascii=False)

    keep = list(full)
    while keep:
        omitted = len(full) - len(keep)
        # Cost of everything except the passage text itself -- metadata,
        # citations, diagnostics, JSON punctuation. This is the part that must
        # survive, so it is measured first and never traded away.
        overhead = len(render([dict(r, text="") for r in keep], omitted))
        room = char_budget - overhead
        if room >= MIN_EXCERPT_CHARS * len(keep):
            # JSON escaping (newlines, quotes) inflates the rendered size above
            # the raw character count, so the first estimate can overshoot.
            # Shrink the excerpt before sacrificing a whole result -- a shorter
            # quote still carries its citation, whereas a dropped result carries
            # nothing at all.
            per = room // len(keep)
            while per >= MIN_EXCERPT_CHARS:
                trimmed = [dict(r, text=_clip(r["text"], per)) for r in keep]
                out = render(trimmed, omitted)
                if len(out) <= char_budget:
                    return out
                per = int(per * 0.85)
        keep.pop()   # only now: drop the weakest-ranked result and retry

    return render([], len(full))


# -- CLI ----------------------------------------------------------------------

def calibrate(write: bool = True) -> dict:
    """
    Measure where 'relevant' and 'irrelevant' actually land on THIS corpus,
    with THESE models, and write the thresholds to data/calibration.json.

    WHY THIS EXISTS
    ---------------
    Every relevance threshold in this file started life as a plausible-looking
    constant, and every one of them was wrong:

      * The cosine baseline was set to 0.45. Genuinely unrelated text on this
        corpus scores 0.542 -- above it. Unrelated passages read "moderate".
      * The cross-encoder thresholds assumed roughly -11..+11 logits with 0 as
        the neutral point. Against abstract prose the model's whole range sits
        lower: "Nameless Dread" scored -2.71 for a query about dread, an exact
        title match landing below a -2.0 cutoff and being thrown away.

    The failure mode is the same each time: a number that sounds reasonable,
    chosen without measuring. So this measures. It probes with queries that
    MUST match and queries that MUST NOT, reads where each lands, and puts the
    boundaries between them.

    Re-run after changing the embedding model, the reranker, or the corpus.
    """
    _load()

    nonsense = [
        "zebra spacecraft telemetry calibration",
        "quarterly dental insurance reimbursement schedule",
        "carburettor gasket torque specification",
        "recipe for lemon drizzle cake",
    ]
    genuine = [
        "the foreclosure of political alternatives under capitalism",
        "how the historian selects which facts become history",
        "cattle and the reckoning of time",
        "what have I written about dread",
    ]

    reranker = _get_reranker()
    using_ce = reranker is not None

    def probe(q):
        """Top score for q under the scorer that will actually be used."""
        dense_order, sims, dist = _dense_ranking(q, None)
        sparse_order = _sparse_ranking(q, None)
        fused = defaultdict(float)
        for r, i in enumerate(dense_order):
            fused[i] += 1.0 / (RRF_K + r + 1)
        for r, i in enumerate(sparse_order):
            fused[i] += 1.0 / (RRF_K + r + 1)
        cands = sorted(fused, key=lambda i: fused[i], reverse=True)[:RERANK_POOL]
        if not cands:
            return None, None, None
        top_sim = max((sims.get(i, 0.0) for i in cands), default=0.0)
        if using_ce:
            scores = reranker.predict([(q, _docs[i]) for i in cands])
            best = max(float(s) for s in scores)
            best_i = cands[int(max(range(len(scores)), key=lambda k: scores[k]))]
        else:
            scored = [(i, _calibrated_score(sims[i], dist)) for i in cands if i in sims]
            if not scored:
                return None, None, top_sim
            best_i, best = max(scored, key=lambda t: t[1])
        return best, _metas[best_i].get("title", "?"), top_sim

    scorer = "cross-encoder" if using_ce else "calibrated cosine"
    print(f"\nCorpus  : {len(_docs)} chunks")
    print(f"Embedder: {embeddings.EMBED_MODEL}")
    print(f"Scorer  : {scorer}")
    if not using_ce:
        print(f"          (reranker unavailable: {_rerank_unavailable_reason})")
    print()
    print(f"{'query':<46} {'score':>8} {'cosine':>7}  best match")
    print("-" * 92)

    noise_scores, real_scores = [], []
    for q in nonsense:
        s, title, sim = probe(q)
        if s is None:
            continue
        noise_scores.append(s)
        print(f"{('[noise] ' + q)[:46]:<46} {s:>8.2f} {sim:>7.3f}  {str(title)[:28]}")
    for q in genuine:
        s, title, sim = probe(q)
        if s is None:
            continue
        real_scores.append(s)
        print(f"{('[real]  ' + q)[:46]:<46} {s:>8.2f} {sim:>7.3f}  {str(title)[:28]}")

    if not noise_scores or not real_scores:
        print("\nNot enough signal to calibrate.")
        return {}

    hi_noise, lo_real = max(noise_scores), min(real_scores)
    gap = lo_real - hi_noise
    print(f"\n  worst genuine query scored : {lo_real:+.2f}")
    print(f"  best nonsense query scored : {hi_noise:+.2f}")
    print(f"  separation                 : {gap:+.2f}")

    if gap <= 0:
        print("\n  PROBLEM: nonsense scores as high as real queries. The scorer")
        print("  cannot tell them apart on this corpus, so no threshold will")
        print("  work. Check the index first:  python3 ingest.py --report")
        return {}

    # ALL THREE boundaries go INSIDE the measured gap, above the noise ceiling.
    #
    # The floor is the subtle one. Stage 3 uses `moderate` as the normal cutoff,
    # but relaxes down to `floor` when the source-diversity cap has starved the
    # result set. So the floor is the last line of defence -- and if it sits
    # BELOW where nonsense scores, nonsense gets rescued by that relaxation and
    # "nothing in the corpus is relevant" quietly stops being reachable.
    #
    # A first version of this put the floor at hi_noise - gap*0.25, i.e. below
    # the noise, on the reasoning that a floor should be generous. Backwards:
    # the floor's whole job is to be the thing nonsense cannot clear.
    floor = round(hi_noise + gap * 0.12, 2)     # just above the worst nonsense
    moderate = round(hi_noise + gap * 0.35, 2)  # normal cutoff
    strong = round(hi_noise + gap * 0.75, 2)    # confident enough to build on

    cal = {
        "scorer": scorer,
        "corpus_chunks": len(_docs),
        "embed_model": embeddings.EMBED_MODEL,
        "strong": strong,
        "moderate": moderate,
        "floor": floor,
        "measured_noise_ceiling": round(hi_noise, 3),
        "measured_real_floor": round(lo_real, 3),
    }

    print(f"\n  strong   >= {strong:+.2f}")
    print(f"  moderate >= {moderate:+.2f}")
    print(f"  floor     < {floor:+.2f}   (below this, nothing is returned)")

    # Verify rather than assume. Both of these are the properties the whole
    # calibration exists to guarantee, so check them explicitly and say so.
    noise_blocked = hi_noise < floor
    real_kept = lo_real >= moderate
    print(f"\n  check: worst nonsense ({hi_noise:+.2f}) below the floor?  "
          f"{'yes' if noise_blocked else 'NO -- nonsense can still get through'}")
    print(f"  check: worst genuine  ({lo_real:+.2f}) above moderate?  "
          f"{'yes' if real_kept else 'NO -- real matches will be dropped'}")
    if not (noise_blocked and real_kept):
        print("\n  Refusing to write a calibration that fails its own checks.")
        return {}

    if write:
        os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
        with open(CALIBRATION_PATH, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"\n  Written to {CALIBRATION_PATH}")
        print("  These now override the defaults in corpus.py automatically.\n")
    return cal


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
        sys.exit(0)

    _load()
    if os.path.exists(CALIBRATION_PATH):
        print(f"(using measured thresholds: strong>={STRONG:+.2f} "
              f"moderate>={MODERATE:+.2f} floor<{FLOOR:+.2f})")
    else:
        print("(!) Thresholds are UNCALIBRATED guesses. "
              "Run: python3 corpus.py --calibrate")

    q = " ".join(sys.argv[1:]) or "alienation and modern society"
    out = search_corpus(q, n_results=5)
    d = out["diagnostics"]
    print(f"\nQuery   : {q}")
    print(f"Verdict : {d['verdict']}")
    print(f"Scoring : {d['scoring']}  |  corpus: {d['corpus_chunks']} chunks  "
          f"|  dense {d['dense_hits']} / sparse {d['sparse_hits']} -> "
          f"{d['fused_candidates']} candidates")
    if d.get("reranker_note"):
        print(f"NOTE    : reranker unavailable -- {d['reranker_note']}")
    print()
    for i, r in enumerate(out["results"], 1):
        print(f"[{i}] {r['citation']}")
        print(f"    relevance {r['relevance']} ({r['confidence']}), "
              f"cosine {r['similarity']}, found by {r['found_by']}")
        print(f"    {r['text'][:260]}...\n")


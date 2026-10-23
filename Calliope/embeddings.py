"""
Calliope Embedding Layer
========================

WHAT AN EMBEDDING IS (read this once, then never again)
-------------------------------------------------------
An embedding model turns a piece of text into a list of numbers -- a vector.
Think of it as a coordinate in a very high-dimensional space (768 dimensions
here). The model is trained so that text with similar *meaning* lands in
similar places, even when the words are totally different.

    "the boss fired him"        -> [0.12, -0.44, 0.91, ...]
    "he was let go by his employer" -> [0.13, -0.41, 0.89, ...]   (close by)
    "the soup was cold"         -> [-0.77, 0.05, -0.22, ...]      (far away)

"Semantic search" is then just: embed the query, and find the stored chunks
whose vectors are closest. Closeness is measured by cosine similarity -- the
angle between the two vectors. 1.0 = identical direction, 0.0 = unrelated.

So the honest answer to "is it just matching semantic globs?" is: yes, kind of.
It is a nearest-neighbour lookup in a geometric space. There is no reading,
no comprehension, no argument-following. The *quality* of that space is
therefore everything, which is why the choice of model below matters.

WHY WE CHANGED MODELS
---------------------
The old setup used `all-MiniLM-L6-v2`. It is fast and small, but it has a hard
input limit of 256 wordpiece tokens. Calliope's chunks were 400 WORDS, which is
roughly 520-560 tokens. MiniLM silently truncated every chunk at ~45% and
embedded only the first part. Half the corpus was never actually searchable --
it was stored, but its meaning was never encoded.

`nomic-embed-text-v1.5` has an 8192-token limit, 768 dimensions, and scores
far better on retrieval benchmarks. It is already installed in LM Studio, so
there is nothing new to download and it runs on the GPU via Metal.

NOMIC'S TASK PREFIXES -- DO NOT SKIP THESE
------------------------------------------
Nomic is trained asymmetrically. Documents and queries must be prefixed:

    documents -> "search_document: <text>"
    queries   -> "search_query: <text>"

If you embed both sides the same way, retrieval quality drops noticeably.
This module handles the prefixing for you; callers just say which side they're
on. The prefix is baked into the stored vectors, so if you ever change it you
must re-ingest.
"""

from __future__ import annotations

import os
import sys

# -- Config -------------------------------------------------------------------

LM_STUDIO_URL = os.environ.get("CALLIOPE_LM_URL", "http://localhost:1234/v1")
EMBED_MODEL = os.environ.get("CALLIOPE_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")

# Bumped whenever the embedding model or prefix scheme changes. Chroma stores
# fixed-width vectors, so mixing 384-dim MiniLM vectors with 768-dim Nomic
# vectors in one collection is not just wrong, it is a hard error. Changing
# this constant forces a clean, separate index.
EMBED_SPACE_VERSION = "nomic-v1.5"

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Embeddings need the openai package: pip3 install openai")
        _client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    return _client


def _prefixed(texts: list[str], kind: str) -> list[str]:
    """Apply Nomic's asymmetric task prefix."""
    if kind == "document":
        return [f"search_document: {t}" for t in texts]
    if kind == "query":
        return [f"search_query: {t}" for t in texts]
    raise ValueError(f"kind must be 'document' or 'query', got {kind!r}")


def embed(texts: list[str], kind: str, batch_size: int = 16,
          progress: bool = False) -> list[list[float]]:
    """
    Embed a list of strings.

    kind: "document" for corpus chunks being indexed,
          "query"    for a user/Calliope search string.

    Batched because LM Studio will happily accept a huge list and then stall
    for a very long time with no output. Small batches keep it responsive and
    let us show progress on a long ingest.
    """
    if not texts:
        return []

    client = _get_client()
    out: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = _prefixed(texts[i:i + batch_size], kind)
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        except Exception as e:
            raise RuntimeError(
                f"Embedding call failed against {LM_STUDIO_URL} using model "
                f"'{EMBED_MODEL}'.\n"
                f"  - Is LM Studio running with its local server enabled?\n"
                f"  - Is the embedding model downloaded? (LM Studio > Search > "
                f"'nomic-embed-text-v1.5')\n"
                f"Underlying error: {e}"
            ) from e

        # LM Studio does not guarantee ordering, so sort by index defensively.
        for item in sorted(resp.data, key=lambda d: d.index):
            out.append(list(item.embedding))

        if progress:
            done = min(i + batch_size, len(texts))
            pct = 100 * done / len(texts)
            print(f"\r    embedding {done}/{len(texts)} ({pct:.0f}%)",
                  end="", flush=True)

    if progress:
        print()

    return out


def embed_one(text: str, kind: str) -> list[float]:
    return embed([text], kind)[0]


def dimension() -> int:
    """Probe the live embedding dimension. Useful for sanity checks."""
    return len(embed_one("dimension probe", "query"))


if __name__ == "__main__":
    print(f"Endpoint : {LM_STUDIO_URL}")
    print(f"Model    : {EMBED_MODEL}")
    try:
        d = dimension()
    except RuntimeError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
    print(f"Dimension: {d}")

    # Show the geometry doing its job.
    import math
    a, b, c = embed([
        "capitalism forecloses the political imagination",
        "it is easier to imagine the end of the world than the end of capitalism",
        "the cattle are herded into the byre at dusk",
    ], "document")

    def cos(u, v):
        dot = sum(x * y for x, y in zip(u, v))
        nu = math.sqrt(sum(x * x for x in u))
        nv = math.sqrt(sum(x * x for x in v))
        return dot / (nu * nv)

    print(f"\nrelated   pair similarity: {cos(a, b):.3f}   (should be high)")
    print(f"unrelated pair similarity: {cos(a, c):.3f}   (should be lower)")

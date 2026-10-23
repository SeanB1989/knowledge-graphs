"""
Calliope Graph Search
=====================

Semantic search over the ontology graph, using the same embedding space the
corpus already uses.

WHY THIS EXISTS
---------------
`CalliopeGraph.search()` in v1 was a substring match:

    searchable = node.lower() + " " + " ".join(str(v).lower() for v in attrs.values())
    if query_lower in searchable:

Its docstring called this "fuzzy search". It is not fuzzy. It is `in`. The
consequence was that the graph could only be found by literal character
overlap, so:

    search("honesty")             -> found "reasoning integrity" BY LUCK,
                                     because the word "honestly" happens to
                                     appear in its description
    search("intellectual courage") -> nothing
    search("am I bullshitting myself") -> nothing

Meanwhile the corpus half of Calliope has been doing proper semantic retrieval
this whole time -- Nomic embeddings, cosine similarity, calibrated thresholds.
The graph got the dumb one. This module closes that gap, so a question about
meaning finds nodes that share meaning rather than nodes that share letters.

HOW IT WORKS
------------
1. Every node is rendered to a single line of text: "id (Type). description"
2. That line is embedded once with Nomic ("search_document:" prefix) and
   cached on disk next to the graph.
3. A query is embedded with the "search_query:" prefix and compared to every
   cached vector by cosine similarity.
4. Results above a MEASURED floor are returned, best first.

The cache is keyed by a hash of each node's text, so editing one description
re-embeds one node, not all 54. Nodes deleted from the graph fall out of the
cache on the next write.

WHY BRUTE FORCE IS CORRECT HERE
-------------------------------
54 nodes x 768 dimensions is about 41,000 multiply-adds per query -- roughly a
tenth of a millisecond. An approximate index (HNSW, FAISS) exists to avoid
scanning millions of vectors and pays for that with build time, memory, and
recall you can't fully trust. At this scale exact search is both faster and
perfect. Do not "optimise" this until the graph has six figures of nodes,
which is to say: never.

GRACEFUL DEGRADATION
--------------------
If LM Studio is not running, embedding raises. The graph must not become
unsearchable because the server is down, so `search()` catches that and falls
back to the old substring behaviour, flagging the result with mode="lexical"
so the caller can tell the difference. Silent degradation is how you end up
debugging retrieval quality for a week; this one announces itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import embeddings

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "data", "graph_embeddings.json")

# How many hits to return by default. The graph is small and Calliope's context
# is 16k, so a generous top-k is cheap; the floor below does the real filtering.
TOP_K = 8

# Absolute similarity floor. MEASURED against the live 54-node graph by
# `python3 graph_search.py --calibrate`, not guessed. See that function for the
# probe methodology and the numbers it produced.
#
# Nomic cosine similarities do not span 0..1 in practice -- unrelated text on
# this graph still scores around 0.35-0.45 because everything shares the
# register of abstract first-person philosophical prose. A naive 0.5 threshold
# would therefore reject correct answers. The value here is set from the
# measured gap between nonsense probes and genuine ones.
#
# MEASURED 2026-08-13 against the live 54-node graph:
#     noise ceiling 0.509  (best score any nonsense probe achieved)
#     real  floor   0.572  (worst score any genuine probe got)
#     midpoint      0.54   <- this value
#
# Note the placeholder before calibration was 0.50, which sat BELOW the noise
# ceiling: "how to descale a kettle" would have returned a node. Measure, then.
#
# The gap is real but narrow (0.063), and it is narrow for a structural reason:
# every node here is abstract first-person philosophical prose, so even a
# carburettor query finds something that rhymes with it tonally. If you start
# seeing genuine queries return nothing, lower this rather than raising it --
# a missed node is invisible to Calliope, whereas a spurious one she can judge
# and discard. Re-run --calibrate after any substantial batch of new nodes.
MIN_SIM = 0.54

# Relative floor: drop anything that scores far below the best hit for this
# query, even if it clears MIN_SIM. Absolute thresholds are fragile across
# query phrasings; a relative one adapts to each query's own distribution.
# 0.86 keeps the tight cluster around the winner and cuts the long tail.
REL_FLOOR = 0.86

INDEX_VERSION = 2  # bump to force a full re-embed


# -- Node -> text -------------------------------------------------------------

# A node's TYPE encodes Sean's stance toward it, but the bare token does not
# survive embedding. v1 rendered nodes as "predictable slop (Anti). The
# aesthetic failure mode: work that is pre-digested..." -- a neutral
# description of a phenomenon, with "(Anti)" carrying almost no semantic
# weight. The measured consequence: the query "what do I refuse to become"
# returned ZERO Anti nodes, despite the Anti category being a literal answer to
# that question. It returned Values and Persons instead.
#
# Rendering the type as a sentence about Sean's relationship to the node puts
# the stance INSIDE the embedded meaning, where a query about stance can reach
# it. It also separates influences from commitments, which is the other thing
# v1 could not do: every Person description is a dense conceptual summary, so
# thinkers competed with values on abstract queries and often won.
STANCE = {
    "Value":     "A value Sean holds -- a commitment he tries to live by.",
    "Anti":      "An anti-value: something Sean rejects, refuses to become, and defines himself against.",
    "Belief":    "A belief Sean holds to be true about how things are.",
    "Concept":   "A concept Sean thinks with.",
    "Person":    "A person whose work has influenced Sean's thinking.",
    "Project":   "A project Sean is building.",
    "Goal":      "A goal Sean is working toward.",
    "Tension":   "A tension: two things Sean wants that pull against each other.",
    "Role":      "A role Sean occupies.",
    "Passage":   "A passage from a text Sean has read.",
    "Text":      "A text in Sean's corpus.",
}


def node_text(node_id: str, attrs: dict) -> str:
    """
    Render a node to the string that gets embedded.

    Shape: "<id>. <stance sentence> <description>"

    The stance sentence is the load-bearing part -- see STANCE above for the
    measurement that forced it. Changing anything here changes every node's
    hash, so the next search re-embeds the whole graph automatically. That is
    intended; re-run --calibrate afterwards, because the similarity
    distribution moves when the document text moves.
    """
    node_type = attrs.get("type", "")
    desc = attrs.get("description", "")
    stance = STANCE.get(node_type) or (f"({node_type})." if node_type else "")
    return " ".join(p for p in (f"{node_id}.", stance, desc) if p).strip()


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# -- Index persistence --------------------------------------------------------

def _load_index() -> dict:
    if not os.path.exists(INDEX_PATH):
        return {}
    try:
        with open(INDEX_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}   # a corrupt cache is a rebuild, not a crash
    if (data.get("version") != INDEX_VERSION
            or data.get("space") != embeddings.EMBED_SPACE_VERSION
            or data.get("model") != embeddings.EMBED_MODEL):
        return {}   # different embedding space -- old vectors are meaningless
    return data.get("vectors", {})


def _save_index(vectors: dict) -> None:
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    payload = {
        "version": INDEX_VERSION,
        "space": embeddings.EMBED_SPACE_VERSION,
        "model": embeddings.EMBED_MODEL,
        "vectors": vectors,
    }
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, INDEX_PATH)   # atomic; never leaves a half-written cache


# -- Vector maths -------------------------------------------------------------

def _normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def _dot(u: list[float], v: list[float]) -> float:
    return sum(x * y for x, y in zip(u, v))


# -- Index maintenance --------------------------------------------------------

def ensure_index(graph, verbose: bool = False) -> dict:
    """
    Bring the on-disk vector cache in line with the current graph.

    Embeds only nodes that are new or whose text has changed, drops nodes that
    no longer exist, and returns {node_id: {"hash", "vec"}} with vectors
    already normalised so search is a plain dot product.

    Raises RuntimeError (from embeddings.embed) if LM Studio is unreachable.
    Callers that need to survive that should use search(), which handles it.
    """
    cached = _load_index()
    current = {n: node_text(n, graph.G.nodes[n]) for n in graph.G.nodes}

    stale = [n for n, t in current.items() if cached.get(n, {}).get("hash") != _hash(t)]
    dropped = [n for n in cached if n not in current]

    if stale:
        if verbose:
            print(f"  embedding {len(stale)} node(s): {', '.join(stale[:5])}"
                  f"{' ...' if len(stale) > 5 else ''}")
        vecs = embeddings.embed([current[n] for n in stale], "document")
        for n, v in zip(stale, vecs):
            cached[n] = {"hash": _hash(current[n]), "vec": _normalise(v)}

    for n in dropped:
        cached.pop(n, None)

    if stale or dropped:
        _save_index(cached)
    elif verbose:
        print("  index already current")

    return cached


# -- Search -------------------------------------------------------------------

def lexical_search(graph, query: str, node_type: str | None = None) -> list:
    """The original v1 substring match. Kept as the fallback path."""
    q = query.lower()
    out = []
    for node in graph.G.nodes:
        attrs = graph.G.nodes[node]
        if node_type and attrs.get("type") != node_type:
            continue
        hay = node.lower() + " " + " ".join(str(v).lower() for v in attrs.values())
        if q in hay:
            out.append({"id": node, **attrs})
    return out


def semantic_search(graph, query: str, top_k: int = TOP_K,
                    min_sim: float = MIN_SIM,
                    node_type: str | None = None) -> list:
    """
    Rank every node against the query by cosine similarity.

    A node whose id contains the query literally is always included regardless
    of its vector score. Exact-name lookup is a different intent from
    conceptual search, and it would be perverse for search("channeling") to
    miss the node called "channeling" because the embedding disagreed.

    node_type restricts the search to one category. This exists because of a
    real result from --calibrate: the query "intellectual courage" returned
    Michel Houellebecq (a Person) above every Value, including "character
    virtues", whose description literally lists courage. Person nodes here
    carry dense conceptual descriptions of what each thinker REPRESENTS, so
    they compete directly with Values and Concepts on abstract queries. That
    is often the right answer -- but when Calliope wants Sean's commitments
    rather than his influences, she needs to be able to say so. The relative
    floor is computed within the filtered set, so filtering does not drag in
    weak hits that would have lost to a Person.
    """
    index = ensure_index(graph)
    if not index:
        return []

    qvec = _normalise(embeddings.embed_one(query, "query"))
    q = query.lower()

    scored = []
    for node_id, entry in index.items():
        if node_id not in graph.G:
            continue
        if node_type and graph.G.nodes[node_id].get("type") != node_type:
            continue
        sim = _dot(qvec, entry["vec"])
        exact = q in node_id.lower()
        scored.append((sim, exact, node_id))

    scored.sort(key=lambda t: (t[1], t[0]), reverse=True)

    best = max((s for s, _, _ in scored), default=0.0)
    rel_cut = best * REL_FLOOR

    out = []
    for sim, exact, node_id in scored:
        if not exact and (sim < min_sim or sim < rel_cut):
            continue
        attrs = dict(graph.G.nodes[node_id])
        out.append({
            "id": node_id,
            **attrs,
            "similarity": round(sim, 4),
            "matched": "name" if exact else "meaning",
        })
        if len(out) >= top_k:
            break
    return out


# Similarity at which a proposed new node is worth flagging as a possible
# duplicate of one that already exists. Set below the measured near-paraphrase
# level (~0.74 for "knowing something in the body rather than the head" against
# "somatic knowing") so genuine restatements get caught, and deliberately kept
# as a WARNING rather than a block: a close-but-distinct node is a legitimate
# thing to add, and silently refusing one would be invisible from the outside.
DUPLICATE_WARN_AT = 0.75


def find_similar_node(graph, node_id: str, node_type: str, description: str,
                      threshold: float = DUPLICATE_WARN_AT, top_k: int = 3) -> list:
    """
    Find existing nodes that may already say what a proposed node would say.

    Called by assert_node BEFORE the node is added, so the index contains only
    pre-existing nodes and the candidate cannot match itself.

    Both sides are embedded as DOCUMENTS rather than one as a query: this is a
    document-to-document comparison ("do these two entries mean the same
    thing?"), not a retrieval ("which entry answers this question?"), and
    Nomic's asymmetric prefixes would skew it if mixed.
    """
    index = ensure_index(graph)
    if not index:
        return []

    text = node_text(node_id, {"type": node_type, "description": description})
    vec = _normalise(embeddings.embed_one(text, "document"))

    key = " ".join(node_id.split()).lower()
    out = []
    for nid, entry in index.items():
        if nid not in graph.G or " ".join(nid.split()).lower() == key:
            continue
        sim = _dot(vec, entry["vec"])
        if sim >= threshold:
            attrs = graph.G.nodes[nid]
            out.append({
                "id": nid,
                "type": attrs.get("type"),
                "similarity": round(sim, 4),
                "description": (attrs.get("description") or "")[:160],
            })
    out.sort(key=lambda d: -d["similarity"])
    return out[:top_k]


def search(graph, query: str, top_k: int = TOP_K,
           node_type: str | None = None) -> dict:
    """
    Public entry point. Always returns:

        {"mode": "semantic"|"lexical", "query": ..., "results": [...]}

    mode="lexical" means the embedding server was unreachable and this was a
    substring match -- degraded, but not silently so.
    """
    try:
        results = semantic_search(graph, query, top_k=top_k, node_type=node_type)
        return {"mode": "semantic", "query": query, "results": results}
    except Exception as e:
        return {
            "mode": "lexical",
            "query": query,
            "results": lexical_search(graph, query, node_type=node_type),
            "warning": f"semantic search unavailable, fell back to substring match: {e}",
        }


# -- Calibration --------------------------------------------------------------

def calibrate(graph) -> dict:
    """
    Measure, don't guess.

    corpus.py learned this the hard way: thresholds that look sane in the
    abstract reject obviously-correct matches, because a given corpus occupies
    its own band of the similarity range. The same applies here, more so --
    every node in this graph is abstract first-person philosophical prose, so
    even unrelated pairs score higher than intuition suggests.

    Probes the graph with queries that SHOULD match nothing and queries that
    SHOULD match something, then reports where each actually lands.
    """
    nonsense = [
        "carburettor gasket torque specification",
        "recipe for lemon drizzle cake",
        "flight departure times to Lisbon",
        "how to descale a kettle",
    ]
    genuine = [
        "honesty",
        "am I bullshitting myself",
        "intellectual courage",
        "thinking clearly without flattering myself",
        "what do I refuse to become",
        "knowing something in the body rather than the head",
    ]

    index = ensure_index(graph, verbose=True)
    if not index:
        raise RuntimeError("no index")

    def ranked_for(q):
        qv = _normalise(embeddings.embed_one(q, "query"))
        scores = [(_dot(qv, e["vec"]), n) for n, e in index.items()]
        scores.sort(reverse=True)
        return scores

    def label(n):
        t = graph.G.nodes[n].get("type", "?")
        return f"{n} ({t})"

    print("\nNONSENSE probes (these should score LOW):")
    noise = []
    for q in nonsense:
        s, n = ranked_for(q)[0]
        noise.append(s)
        print(f"  {s:.3f}  {q!r} -> {label(n)}")

    print("\nGENUINE probes (these should score HIGH):")
    real = []
    spreads = []
    for q in genuine:
        r = ranked_for(q)
        s, n = r[0]
        real.append(s)
        # Discrimination matters more than magnitude. Cosine values live in a
        # compressed band on this graph (every node is abstract first-person
        # prose), so a "low" 0.57 is not a problem -- but a best-to-8th spread
        # of 0.03 is, because it means the ordering below rank 2 is noise.
        eighth = r[min(7, len(r) - 1)][0]
        spreads.append(s - eighth)
        print(f"  {s:.3f}  (spread to 8th: {s - eighth:.3f})  {q!r} -> {label(n)}")

    print("\nDISCRIMINATION (gap between the best hit and the 8th):")
    avg = sum(spreads) / len(spreads)
    print(f"  mean spread: {avg:.3f}")
    if avg < 0.04:
        print("  POOR. The ranking below rank 2 is close to arbitrary. Enrich the")
        print("  document text (see STANCE) rather than tuning thresholds.")
    elif avg < 0.08:
        print("  FAIR. Top few hits are meaningful; the tail is not. Consider")
        print(f"  tightening REL_FLOOR (currently {REL_FLOOR}) to cut it.")
    else:
        print("  GOOD. The space is separating these queries cleanly.")

    ceiling = max(noise)
    floor = min(real)
    print(f"\n  noise ceiling : {ceiling:.3f}  (best score any nonsense achieved)")
    print(f"  real floor    : {floor:.3f}  (worst score any genuine query got)")
    if floor > ceiling:
        suggested = round((ceiling + floor) / 2, 2)
        print(f"  GAP IS CLEAN. Suggested MIN_SIM = {suggested}  (currently {MIN_SIM})")
    else:
        print(f"  WARNING: bands overlap. No single threshold separates them; "
              f"lean on REL_FLOOR (currently {REL_FLOOR}) instead.")
    return {"noise_ceiling": ceiling, "real_floor": floor}


if __name__ == "__main__":
    import sys
    from graph import CalliopeGraph

    g = CalliopeGraph()
    if "--calibrate" in sys.argv:
        calibrate(g)
    elif "--reindex" in sys.argv:
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        ensure_index(g, verbose=True)
        print("done")
    else:
        args = sys.argv[1:]
        ntype = None
        if "--type" in args:
            i = args.index("--type")
            ntype = args[i + 1]
            del args[i:i + 2]
        q = " ".join(a for a in args if not a.startswith("--")) or "honesty"
        r = search(g, q, node_type=ntype)
        print(f"[{r['mode']}] {q!r}")
        if r.get("warning"):
            print(f"  {r['warning']}")
        for hit in r["results"]:
            sim = hit.get("similarity")
            tag = f"{sim:.3f} {hit.get('matched','')}" if sim is not None else "lexical"
            print(f"  {tag:>16}  {hit['id']} ({hit.get('type')})")

"""
Unit tests for graph_search.py

These run WITHOUT LM Studio and WITHOUT networkx. The embedding layer is
stubbed with a deterministic bag-of-words vectoriser, so cosine similarity
becomes plain word overlap and ranking assertions are exact rather than
vibes-based. What this proves is the plumbing: cache invalidation, atomic
writes, exact-name override, relative floor, and the fallback path.

What it deliberately does NOT prove is the quality of Nomic's semantic space.
That is measured, not asserted -- run `python3 graph_search.py --calibrate`
against the live graph for that.

    python3 test_graph_search.py
"""
import json, math, os, sys, tempfile, types

# -- stub the embedding layer BEFORE graph_search imports it ------------------
stub = types.ModuleType("embeddings")
stub.EMBED_SPACE_VERSION = "test-space"
stub.EMBED_MODEL = "test-model"
stub.calls = []          # records every batch handed to embed()
stub.fail = False        # flip to simulate LM Studio being down

_VOCAB = ["honest", "rigour", "body", "craft", "noise", "money", "cake", "value", "concept"]

def _vec(text):
    t = text.lower()
    v = [float(t.count(w)) for w in _VOCAB]
    if not any(v):
        v = [0.01] * len(_VOCAB)
    return v

def embed(texts, kind, **kw):
    if stub.fail:
        raise RuntimeError("simulated: LM Studio unreachable")
    stub.calls.append(list(texts))
    return [_vec(t) for t in texts]

def embed_one(text, kind):
    return embed([text], kind)[0]

stub.embed = embed
stub.embed_one = embed_one
sys.modules["embeddings"] = stub

import graph_search as gs

# -- a fake graph that quacks like CalliopeGraph ------------------------------
class FakeG:
    def __init__(self, nodes): self.nodes = dict(nodes)
    def __contains__(self, n): return n in self.nodes

class FakeGraph:
    def __init__(self, nodes): self.G = FakeG(nodes)

def sample():
    return FakeGraph({
        "reasoning integrity": {"type": "Value",
            "description": "honest honest rigour rigour thinking without flattering"},
        "somatic knowing":     {"type": "Belief",
            "description": "body body knowledge felt in the body not the head"},
        "editorial conscience":{"type": "Value",
            "description": "craft craft craft refusing noise"},
        "optimization culture":{"type": "Anti",
            "description": "money money noise metrics over meaning"},
    })

PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok   {name}")
    else:    FAIL += 1; print(f"  FAIL {name} {extra}")

def fresh_index_path():
    fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd); os.unlink(p)
    gs.INDEX_PATH = p
    return p

print("graph_search unit tests\n")

# 1 -- node_text
g = sample()
txt = gs.node_text("reasoning integrity", g.G.nodes["reasoning integrity"])
check("node_text leads with id, then stance, then description",
      txt.startswith("reasoning integrity. A value Sean holds") and "flattering" in txt, txt)
check("Anti nodes render as an explicit refusal",
      "refuses to become" in gs.node_text("optimization culture", {"type": "Anti", "description": "x"}),
      gs.node_text("optimization culture", {"type": "Anti", "description": "x"}))
check("unknown type still renders without crashing",
      gs.node_text("mystery", {"type": "Wat", "description": "y"}).startswith("mystery. (Wat)."),
      gs.node_text("mystery", {"type": "Wat", "description": "y"}))

# 2 -- first build embeds every node exactly once
p = fresh_index_path(); stub.calls.clear()
idx = gs.ensure_index(g)
check("first build embeds all 4 nodes", len(idx) == 4 and len(stub.calls[0]) == 4,
      f"{len(idx)} nodes, batches={[len(c) for c in stub.calls]}")
check("vectors are normalised",
      abs(math.sqrt(sum(x*x for x in idx["somatic knowing"]["vec"])) - 1.0) < 1e-9)

# 3 -- second build embeds nothing
stub.calls.clear(); gs.ensure_index(g)
check("unchanged graph re-embeds nothing", stub.calls == [], stub.calls)

# 4 -- editing one description re-embeds exactly one node
stub.calls.clear()
g.G.nodes["editorial conscience"]["description"] = "craft craft craft and also honest"
gs.ensure_index(g)
check("edited node re-embeds alone",
      len(stub.calls) == 1 and len(stub.calls[0]) == 1 and "editorial conscience" in stub.calls[0][0],
      stub.calls)

# 5 -- deleting a node drops it from the cache
del g.G.nodes["optimization culture"]
idx = gs.ensure_index(g)
check("deleted node leaves the cache", "optimization culture" not in idx and len(idx) == 3)

# 6 -- changing the embedding space invalidates everything
stub.EMBED_SPACE_VERSION = "different-space"; stub.calls.clear()
gs.ensure_index(g)
check("changed embedding space forces full re-embed",
      len(stub.calls) == 1 and len(stub.calls[0]) == 3, stub.calls)
stub.EMBED_SPACE_VERSION = "test-space"

# 7 -- a corrupt cache rebuilds instead of crashing
open(gs.INDEX_PATH, "w").write("{not json at all")
stub.calls.clear()
try:
    idx = gs.ensure_index(g); ok = len(idx) == 3
except Exception as e:
    ok = False; idx = e
check("corrupt cache file rebuilds silently", ok, idx)

# 8 -- no .tmp file survives a write
check("atomic write leaves no .tmp behind", not os.path.exists(gs.INDEX_PATH + ".tmp"))

# 9 -- ranking: a 'body' query should put somatic knowing first
p = fresh_index_path(); gs.ensure_index(g)
r = gs.search(g, "body")
check("semantic mode reported", r["mode"] == "semantic", r["mode"])
check("best hit for 'body' is somatic knowing",
      r["results"] and r["results"][0]["id"] == "somatic knowing",
      [h["id"] for h in r["results"]])
check("hits carry a similarity score",
      all("similarity" in h for h in r["results"]))

# 10 -- exact name match wins even when the vector disagrees
r = gs.search(g, "editorial conscience")
check("exact name match ranked first and tagged",
      r["results"][0]["id"] == "editorial conscience" and r["results"][0]["matched"] == "name",
      r["results"][0] if r["results"] else None)

# 11 -- an unrelated query is filtered out by the floor
r = gs.search(g, "cake")
check("unrelated query returns nothing", r["results"] == [], [h["id"] for h in r["results"]])

# 12 -- LM Studio down => lexical fallback, flagged, still useful
stub.fail = True
r = gs.search(g, "craft")
check("fallback reports mode=lexical", r["mode"] == "lexical", r["mode"])
check("fallback carries a warning", "warning" in r)
check("fallback still finds the substring match",
      any(h["id"] == "editorial conscience" for h in r["results"]),
      [h["id"] for h in r["results"]])
stub.fail = False

# 13-17 -- node_type filter (added after --calibrate showed a Person node
# out-ranking every Value on the query "intellectual courage")
stub.fail = False
g2 = sample()
p = fresh_index_path(); gs.ensure_index(g2)

r = gs.search(g2, "body honest")
check("unfiltered query ranks the Belief node first",
      r["results"][0]["id"] == "somatic knowing", [h["id"] for h in r["results"]])

hits = gs.semantic_search(g2, "body honest", node_type="Value", min_sim=0.3)
ids = [h["id"] for h in hits]
check("type filter excludes other types", "somatic knowing" not in ids, ids)
check("type filter surfaces a hit the unfiltered query buried",
      "reasoning integrity" in ids, ids)
check("filtered results are all the requested type",
      all(h.get("type") == "Value" for h in hits), [h.get("type") for h in hits])

stub.fail = True
r = gs.search(g2, "craft", node_type="Belief")
check("type filter applies on the lexical fallback too",
      r["mode"] == "lexical" and r["results"] == [],
      (r["mode"], [h["id"] for h in r["results"]]))
stub.fail = False

# 18-22 -- find_similar_node (feeds assert_node's duplicate warning)
g3 = sample()
p = fresh_index_path(); gs.ensure_index(g3)

hits = gs.find_similar_node(g3, "bodily knowledge", "Belief",
                            "body body knowledge felt in the body", threshold=0.5)
check("a restatement of an existing node is flagged",
      hits and hits[0]["id"] == "somatic knowing", hits)
check("flagged candidate carries similarity and description",
      hits and "similarity" in hits[0] and "description" in hits[0], hits[:1])

hits = gs.find_similar_node(g3, "lemon cake recipe", "Concept", "cake cake money",
                            threshold=0.5)
check("a genuinely new node is not flagged", hits == [], hits)

hits = gs.find_similar_node(g3, "Somatic Knowing", "Belief",
                            "body body knowledge felt in the body", threshold=0.5)
check("a node never matches itself under different casing",
      all(h["id"] != "somatic knowing" for h in hits), hits)

stub.fail = True
try:
    gs.find_similar_node(g3, "x", "Value", "honest", threshold=0.5)
    ok = False
except Exception:
    ok = True   # raises, so assert_node's try/except is what keeps writes working
check("find_similar_node raises when the embedder is down (caller must catch)", ok)
stub.fail = False

os.path.exists(gs.INDEX_PATH) and os.unlink(gs.INDEX_PATH)
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)

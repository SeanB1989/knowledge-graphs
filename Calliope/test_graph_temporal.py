"""
Unit tests for the temporal layer: assertion history, soft retirement,
and what_changed().

CAVEAT WORTH READING: networkx is stubbed here with a hand-written DiGraph
that implements only the handful of behaviours graph.py actually uses --
attribute-MERGING add_node, add_edge, degree, in/out_edges, removal. It is
faithful for those operations and nothing else. A green run proves Calliope's
history logic is correct; it does NOT prove compatibility with real networkx.

    python3 test_graph_temporal.py
"""
import json, os, sys, tempfile, types

try:
    import networkx  # noqa: F401
except ImportError:
    class _DiGraph:
        def __init__(self): self._n, self._e = {}, {}
        # -- nodes
        @property
        def nodes(self): return _NodeView(self._n)
        def add_node(self, n, **a): self._n.setdefault(n, {}).update(a)   # MERGES, like nx
        def __contains__(self, n): return n in self._n
        def remove_node(self, n):
            self._n.pop(n, None)
            for k in [k for k in self._e if n in k]: del self._e[k]
        # -- edges
        @property
        def edges(self): return _EdgeView(self._e)
        def add_edge(self, u, v, **a):
            self._n.setdefault(u, {}); self._n.setdefault(v, {})
            self._e.setdefault((u, v), {}).update(a)
        def has_edge(self, u, v): return (u, v) in self._e
        def remove_edge(self, u, v): self._e.pop((u, v), None)
        def in_edges(self, n): return [(u, v) for (u, v) in self._e if v == n]
        def out_edges(self, n): return [(u, v) for (u, v) in self._e if u == n]
        def degree(self, n): return sum(1 for k in self._e if n in k)
        def successors(self, n): return [v for (u, v) in self._e if u == n]
        def predecessors(self, n): return [u for (u, v) in self._e if v == n]
        def number_of_nodes(self): return len(self._n)
        def number_of_edges(self): return len(self._e)
    class _NodeView:
        def __init__(self, d): self._d = d
        def __iter__(self): return iter(list(self._d))
        def __getitem__(self, k): return self._d[k]
        def __contains__(self, k): return k in self._d
        def __len__(self): return len(self._d)
    class _EdgeView:
        def __init__(self, d): self._d = d
        def __iter__(self): return iter(list(self._d))
        def __getitem__(self, k): return self._d[tuple(k)] if isinstance(k, tuple) else self._d[k]
        def __len__(self): return len(self._d)
    nx = types.ModuleType("networkx")
    nx.DiGraph = _DiGraph
    nx.NetworkXNoPath = type("NetworkXNoPath", (Exception,), {})
    nx.NodeNotFound = type("NodeNotFound", (Exception,), {})
    nx.shortest_path = lambda *a, **k: []
    sys.modules["networkx"] = nx

import graph as G

PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok   {name}")
    else:    FAIL += 1; print(f"  FAIL {name} {extra}")

def fresh():
    fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd); os.unlink(p)
    return G.CalliopeGraph(path=p)

TODAY = G._today()
print("temporal layer tests\n")

# -- assertion stamps ---------------------------------------------------------
g = fresh()
g.assert_node("reasoning integrity", "Value", "thinking honestly", weight=0.7)
n = g.G.nodes["reasoning integrity"]
check("new node stamped with first_asserted", n.get("first_asserted") == TODAY, n.get("first_asserted"))
check("new node has empty history", n.get("history") == [], n.get("history"))

# -- weight change is recorded ------------------------------------------------
g.assert_node("reasoning integrity", "Value", "thinking honestly", weight=0.9)
n = g.G.nodes["reasoning integrity"]
check("weight change recorded in history",
      any("0.7 -> 0.9" in h["change"] for h in n["history"]), n["history"])
check("first_asserted survives an update", n["first_asserted"] == TODAY)

# -- the old description is PRESERVED, not destroyed --------------------------
g.assert_node("reasoning integrity", "Value", "not rationalising, not flattering", weight=0.9)
n = g.G.nodes["reasoning integrity"]
was = [h.get("was") for h in n["history"] if h["change"] == "description rewritten"]
check("description rewrite keeps the previous text",
      was and was[0] == "thinking honestly", n["history"])

# -- re-asserting identically must NOT spam history ---------------------------
before = len(n["history"])
g.assert_node("reasoning integrity", "Value", "not rationalising, not flattering", weight=0.9)
check("idempotent re-assert adds no history",
      len(g.G.nodes["reasoning integrity"]["history"]) == before, before)

# -- edges get the same treatment ---------------------------------------------
g.assert_node("sycophancy", "Anti", "telling people what they want to hear", weight=0.8)
g.assert_edge("reasoning integrity", "sycophancy", "OPPOSES", weight=0.6, grounding="stated")
g.assert_edge("reasoning integrity", "sycophancy", "OPPOSES", weight=0.95, grounding="stated")
e = g.G.edges["reasoning integrity", "sycophancy"]
check("edge weight change recorded",
      any("0.6 -> 0.95" in h["change"] for h in e["history"]), e.get("history"))
check("edge carries first_asserted", e.get("first_asserted") == TODAY)

# -- soft retirement ----------------------------------------------------------
r = g.remove_edge("reasoning integrity", "sycophancy", reason="decided it was too strong")
check("edge retire reports, not deletes", r.get("retired") and "not deleted" in r.get("note",""))
check("retired edge is out of the live graph",
      not g.G.has_edge("reasoning integrity", "sycophancy"))
check("retired edge is preserved with its reason",
      any(x["kind"] == "edge" and x.get("reason") == "decided it was too strong"
          for x in g.retired), g.retired)

g.assert_node("committee aesthetics", "Anti", "averaging out the sharp edges")
g.assert_edge("committee aesthetics", "sycophancy", "INFORMS", weight=0.5, grounding="x")
r = g.remove_node("committee aesthetics", reason="folded into predictable slop")
check("retiring a node retires its edges too",
      any(x["kind"] == "node" and len(x["edges"]) == 1 for x in g.retired), g.retired)
check("retired node is gone from the live graph", "committee aesthetics" not in g.G)

# -- persistence across reload -------------------------------------------------
g2 = G.CalliopeGraph(path=g.path)
check("retired list survives save/load", len(g2.retired) == len(g.retired),
      (len(g2.retired), len(g.retired)))
check("history survives save/load",
      len(g2.G.nodes["reasoning integrity"]["history"]) == before, )

# -- what_changed --------------------------------------------------------------
w = g2.what_changed()
kinds = {e["event"] for e in w["events"]}
check("what_changed reports assertions", "asserted" in kinds, kinds)
check("what_changed reports retirements", "retired" in kinds, kinds)
check("what_changed reports weight moves",
      any("weight" in k for k in kinds), kinds)
check("events are newest-first",
      [e["date"] for e in w["events"]] == sorted((e["date"] for e in w["events"]), reverse=True))

w2 = g2.what_changed(since="2099-01-01")
check("a future `since` returns nothing", w2["events"] == [], w2["events"])

# -- the honesty requirement: no invented dates for legacy data ---------------
legacy = fresh()
legacy.G.add_node("old value", type="Value", description="from before", weight=0.5)
legacy.assert_node("old value", "Value", "from before", weight=0.8)
n = legacy.G.nodes["old value"]
check("a node with no prior date is marked pre-temporal, not stamped today",
      n["first_asserted"] == G.PRE_TEMPORAL, n["first_asserted"])
w3 = legacy.what_changed()
check("pre-temporal nodes are counted and disclosed",
      w3["undated_nodes"] >= 1 and w3["note"], (w3["undated_nodes"], w3["note"]))
check("pre-temporal nodes are not dated in the event list",
      all(e["event"] != "asserted" or e["what"] != "old value" for e in w3["events"]),
      w3["events"])

# -- symmetric edges: direction must stop hiding relationships ---------------
s = fresh()
for n, t in [("reasoning integrity", "Value"), ("sycophancy", "Anti"),
             ("editorial conscience", "Concept"), ("structure", "Concept")]:
    s.assert_node(n, t, f"desc for {n}", weight=0.7)
# stored attacker -> victim, the way 12 of Sean's 20 OPPOSES edges run
s.assert_edge("sycophancy", "reasoning integrity", "OPPOSES", 0.9, "g")
# a directional edge for contrast
s.assert_edge("editorial conscience", "structure", "INFORMS", 0.6, "g")

out = s.get_neighbors("reasoning integrity", edge_type="OPPOSES", direction="out")
check("symmetric edge is found even asking the 'wrong' way",
      len(out) == 1 and out[0]["node"] == "sycophancy", out)
check("symmetric edge is labelled symmetric, not incoming/outgoing",
      out and out[0]["direction"] == "symmetric", out)

inn = s.get_neighbors("reasoning integrity", edge_type="OPPOSES", direction="in")
check("symmetric edge found from the other side too", len(inn) == 1, inn)
both = s.get_neighbors("reasoning integrity", edge_type="OPPOSES", direction="both")
check("symmetric edge is not double-counted on direction=both", len(both) == 1, both)

d_out = s.get_neighbors("editorial conscience", edge_type="INFORMS", direction="out")
d_in = s.get_neighbors("editorial conscience", edge_type="INFORMS", direction="in")
check("directional edge still respects direction=out", len(d_out) == 1, d_out)
check("directional edge still respects direction=in", d_in == [], d_in)
check("directional edge keeps its outgoing label",
      d_out and d_out[0]["direction"] == "outgoing", d_out)

# -- find_neglected -----------------------------------------------------------
# Give the fixture a real hub, otherwise every node has degree 1 and there is
# nothing for the sort to demonstrate.
s.assert_node("good faith", "Value", "meaning what you say", weight=0.7)
s.assert_node("simulated knowledge", "Anti", "confident-sounding nonsense", weight=0.8)
s.assert_edge("simulated knowledge", "reasoning integrity", "OPPOSES", 0.8, "g")
s.assert_edge("reasoning integrity", "good faith", "REQUIRES", 0.7, "g")

neg = s.find_neglected(limit=10)
check("find_neglected returns every node with degree and last_updated",
      len(neg) == 6 and all("degree" in n and "last_updated" in n for n in neg), neg)
check("least-connected surfaces before the hub",
      [n["id"] for n in neg].index("structure")
      < [n["id"] for n in neg].index("reasoning integrity"),
      [(n["id"], n["degree"]) for n in neg])
check("find_neglected respects limit", len(s.find_neglected(limit=2)) == 2)

os.path.exists(s.path) and os.unlink(s.path)

# -- provenance ---------------------------------------------------------------
p2 = fresh()
p2.assert_node("reasoning integrity", "Value", "honest thinking", weight=0.9,
               provenance="stated")
p2.assert_node("simulated knowledge", "Anti", "confident nonsense", weight=0.8)
check("default provenance is inferred, not stated",
      p2.G.nodes["simulated knowledge"]["provenance"] == "inferred",
      p2.G.nodes["simulated knowledge"].get("provenance"))
check("explicit 'stated' is respected",
      p2.G.nodes["reasoning integrity"]["provenance"] == "stated")

r = p2.assert_node("ego defense", "Concept", "protecting the self-image",
                   provenance="banana")
check("an invalid provenance falls back to inferred, not through",
      p2.G.nodes["ego defense"]["provenance"] == "inferred", r.get("provenance"))

# grounded in a CORPUS chunk -> still inferred (primary evidence)
r = p2.assert_node("social armor", "Concept", "taste as insulation",
                   grounded_in="41f9c2aebd7e4c118a0d6f5b2c9e0a13")
check("grounding in a corpus chunk stays 'inferred'",
      r["provenance"] == "inferred", r["provenance"])
check("an unresolvable ref is reported as external, with a warning",
      r.get("grounded_in_external") and r.get("grounding_note"), r.get("grounding_note"))

# grounded ONLY in other inferences -> derived
r = p2.assert_node("armor as epistemic cost", "Concept", "what the armor costs you",
                   grounded_in="social armor, ego defense")
check("inference resting only on inferences is computed as 'derived'",
      r["provenance"] == "derived", r["provenance"])
check("the derivation is explained back to the caller",
      "provenance_note" in r, r.keys())

# a 'stated' parent does NOT make the child derived
r = p2.assert_node("clarity costs something", "Belief", "seeing clearly is expensive",
                   grounded_in="reasoning integrity, social armor")
check("a stated parent blocks the derived promotion",
      r["provenance"] == "inferred", r["provenance"])

# stated is never downgraded
r = p2.assert_node("plain fact", "Concept", "he said so",
                   provenance="stated", grounded_in="social armor")
check("'stated' is never downgraded to derived", r["provenance"] == "stated")

# edges carry it too
r = p2.assert_edge("social armor", "reasoning integrity", "TENSIONS_WITH", 0.7,
                   "he lives with both", provenance="confirmed",
                   grounded_in="41f9c2aebd7e4c118a0d6f5b2c9e0a13")
check("edges record provenance", r["edge"]["provenance"] == "confirmed", r["edge"])
check("edges record grounded_in as a list",
      r["edge"]["grounded_in"] == ["41f9c2aebd7e4c118a0d6f5b2c9e0a13"], r["edge"])

# provenance changes are history
p2.assert_edge("social armor", "reasoning integrity", "TENSIONS_WITH", 0.7,
               "he lives with both", provenance="stated")
e = p2.G.edges["social armor", "reasoning integrity"]
check("a provenance change is recorded in history",
      any("provenance" in h["change"] for h in e["history"]), e["history"])

# -- assert_passage namespace fix ---------------------------------------------
p3 = fresh()
p3.assert_node("Slavoj Zizek", "Person", "Lacanian Marxism", weight=0.9)
p3.assert_node("simulated knowledge", "Anti", "confident nonsense", weight=0.8)
res = p3.assert_passage(
    passage_id="", text="Ideology is not a dreamlike illusion...",
    author="Slavoj Zizek", title="The Sublime Object of Ideology",
    grounds="simulated knowledge", source="sublime_object_zizek",
    chunk_id="41f9c2aebd7e4c118a0d6f5b2c9e0a13")

check("passage is keyed by the corpus chunk_id",
      res["passage_id"] == "41f9c2aebd7e4c118a0d6f5b2c9e0a13", res["passage_id"])
check("author resolves to the EXISTING person node, no duplicate",
      res["person_node"] == "Slavoj Zizek"
      and len([n for n in p3.G.nodes if p3.G.nodes[n].get("type") == "Person"]) == 1,
      [n for n in p3.G.nodes if p3.G.nodes[n].get("type") == "Person"])
check("Text node carries the corpus source stem as a join key",
      p3.G.nodes[res["text_node"]].get("source") == "sublime_object_zizek",
      p3.G.nodes[res["text_node"]])
check("GROUNDS edge was created to the claim",
      p3.G.has_edge(res["passage_id"], "simulated knowledge")
      and p3.G.edges[res["passage_id"], "simulated knowledge"]["type"] == "GROUNDS")
check("GROUNDS edge is marked inferred -- the reading is Calliope's judgement",
      p3.G.edges[res["passage_id"], "simulated knowledge"]["provenance"] == "inferred")
check("passage node went through assert_node and got stamped",
      "first_asserted" in p3.G.nodes[res["passage_id"]],
      p3.G.nodes[res["passage_id"]].keys())
check("a grounds target that does not exist is reported, not silently dropped",
      "no such node" in str(p3.assert_passage("", "t", "A B", "T",
                                              grounds="nonexistent node")["grounds"]))

for p in (p2.path, p3.path):
    os.path.exists(p) and os.unlink(p)

for p in (g.path, legacy.path):
    os.path.exists(p) and os.unlink(p)
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)

"""
Unit tests for the assert_node safeguards in graph.py:
canonical id resolution, orphan detection, and find_orphans.

networkx is not installed in every environment these tests run in, and none of
the logic under test actually needs a real graph -- only something that answers
`nodes`, `degree`, and `in`. So networkx is stubbed just far enough to import
the module, and the methods are exercised against a hand-built fake. That keeps
the tests honest: they assert Calliope's logic, not networkx's.

    python3 test_graph_nodes.py
"""
import sys, types

if "networkx" not in sys.modules:
    try:
        import networkx  # noqa: F401
    except ImportError:
        nx = types.ModuleType("networkx")
        nx.DiGraph = type("DiGraph", (), {})
        nx.NetworkXNoPath = type("NetworkXNoPath", (Exception,), {})
        nx.NodeNotFound = type("NodeNotFound", (Exception,), {})
        nx.shortest_path = lambda *a, **k: []
        sys.modules["networkx"] = nx

import graph as G

class FakeG:
    def __init__(self, nodes, degrees=None):
        self.nodes = dict(nodes)
        self._deg = dict(degrees or {})
    def __contains__(self, n): return n in self.nodes
    def degree(self, n): return self._deg.get(n, 0)

def obj(nodes, degrees=None):
    o = G.CalliopeGraph.__new__(G.CalliopeGraph)
    o.G = FakeG(nodes, degrees)
    return o

PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok   {name}")
    else:    FAIL += 1; print(f"  FAIL {name} {extra}")

print("graph.py assert_node safeguard tests\n")

NODES = {
    "reasoning integrity": {"type": "Value", "description": "a"},
    "somatic knowing":     {"type": "Belief", "description": "b"},
    "Michel Houellebecq":  {"type": "Person", "description": "c"},
}
g = obj(NODES)

# -- canonical id resolution --------------------------------------------------
check("exact id resolves to itself",
      g._canonical_id("reasoning integrity") == "reasoning integrity")
check("different casing resolves to the existing node",
      g._canonical_id("Reasoning Integrity") == "reasoning integrity",
      g._canonical_id("Reasoning Integrity"))
check("SHOUTING resolves too",
      g._canonical_id("REASONING INTEGRITY") == "reasoning integrity")
check("collapsed whitespace resolves",
      g._canonical_id("reasoning   integrity") == "reasoning integrity")
check("leading/trailing space resolves",
      g._canonical_id("  somatic knowing ") == "somatic knowing")
check("mixed-case Person node resolves to stored casing",
      g._canonical_id("michel houellebecq") == "Michel Houellebecq")
check("a genuinely new id resolves to None",
      g._canonical_id("intellectual courage") is None,
      g._canonical_id("intellectual courage"))
check("near-miss but distinct id is NOT collapsed",
      g._canonical_id("reasoning integrities") is None)

# -- find_orphans -------------------------------------------------------------
g2 = obj(
    {"lonely": {"type": "Concept", "description": "x"},
     "Kieran": {"type": "Person", "description": "wife"},
     "hub":    {"type": "Value", "description": "y"},
     "thread": {"type": "Anti", "description": "z"}},
    {"lonely": 0, "Kieran": 0, "hub": 13, "thread": 1},
)
o = g2.find_orphans()
ids = [n["id"] for n in o]
check("find_orphans() returns only fully unconnected nodes",
      sorted(ids) == ["Kieran", "lonely"], ids)
check("orphan entries carry degree and attrs",
      all("degree" in n and "type" in n for n in o), o)

o1 = [n["id"] for n in g2.find_orphans(max_degree=1)]
check("max_degree=1 also catches single-thread nodes",
      sorted(o1) == ["Kieran", "lonely", "thread"], o1)
check("well-connected nodes are never returned", "hub" not in o1)

o_sorted = g2.find_orphans(max_degree=13)
check("results sort by degree ascending",
      [n["degree"] for n in o_sorted] == sorted(n["degree"] for n in o_sorted),
      [n["degree"] for n in o_sorted])

# -- node type validation still bites ----------------------------------------
check("dropped type Epistemic is now rejected", "Epistemic" not in G.NODE_TYPES)
check("dropped type Aesthetic is now rejected", "Aesthetic" not in G.NODE_TYPES)
check("Belief survived the trim", "Belief" in G.NODE_TYPES)
check("Tension kept as edge-backed schema", "Tension" in G.NODE_TYPES)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)

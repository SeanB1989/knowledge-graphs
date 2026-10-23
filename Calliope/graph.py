"""
Calliope Graph Engine
A live, traversable personal ontology and epistemology graph.
"""

import json
import os
from datetime import date
from typing import Optional
import networkx as nx


def _today() -> str:
    return date.today().isoformat()


# The 54 nodes and 97 edges that existed before the temporal layer landed have
# no known assertion date. Stamping them with today's would be a lie told by the
# very feature meant to record the truth about change, so they get this marker
# instead and what_changed() excludes them from date ranges.
PRE_TEMPORAL = "pre-2026-08-13"


# -- Provenance ----------------------------------------------------------------
# WHERE AN ASSERTION CAME FROM, kept strictly separate from HOW STRONGLY it is
# held (weight) and from HOW WELL A PASSAGE MATCHED A QUERY (retrieval scores).
# Before this, an edge Calliope inferred from a weak corpus hit was byte-identical
# to one Sean dictated, and the graph could not tell you which was which.
#
#   stated     On the record, not worked out: Sean said it in conversation or
#              wrote it, OR it is a verbatim passage/bibliographic fact from the
#              corpus. The common thread is that nobody inferred it.
#   confirmed  Calliope proposed it and Sean approved it.
#   inferred   Calliope concluded it, from corpus evidence or from the graph.
#   derived    Inferred from assertions that are THEMSELVES inferred.
#
# `derived` is computed, never guessed: see _resolve_provenance. That is what
# makes an inference chain visible instead of hypothetical.
PROVENANCE_LEVELS = {"stated", "confirmed", "inferred", "derived"}

# Calliope is the caller, so anything she asserts is her inference until shown
# otherwise. Defaulting to "stated" would let the model launder its own guesses
# into Sean's own voice, which is the failure his graph calls "simulated
# knowledge".
DEFAULT_PROVENANCE = "inferred"


def _refs(grounded_in) -> list:
    """Accept a comma-separated string or a list; return a clean list of ids."""
    if not grounded_in:
        return []
    if isinstance(grounded_in, str):
        return [r.strip() for r in grounded_in.split(",") if r.strip()]
    return [str(r).strip() for r in grounded_in if str(r).strip()]


GRAPH_PATH = os.path.join(os.path.dirname(__file__), "data", "graph.json")

# Default credence for anything asserted without an explicit weight. Matches
# the default advertised in tools.py, which in v1 disagreed with the 1.0 used
# here -- so a node asserted through the tool layer got 0.7 while the same call
# made directly got 1.0.
DEFAULT_WEIGHT = 0.7

# Valid node types.
#
# NOTE: "Belief" was missing here in v1 while five nodes in graph.json already
# used it (somatic knowing, intuitive computation, calibrated fallibility,
# epistemic grief, belief residue). The effect was that assert_node("...",
# "Belief") always failed with "Unknown node type", so Calliope could read
# those nodes but never add to or update that category -- a whole branch of the
# ontology was silently read-only.
# "Epistemic" and "Aesthetic" were removed 2026-08-13. Both had zero instances
# after months of use, and the reason was structural rather than neglect: every
# other type answers "what is my relationship to this?" (Value, Anti, Belief,
# Concept, Person) while those two answered "what domain is this about?" -- a
# different axis smuggled into the same field. So an epistemic value was always
# just filed as a Value, and the slot never got picked.
#
# Sean's ruling: epistemology belongs in the GROUNDING of an edge, not in a node
# type. Where a belief comes from is a relation to a passage, which is what
# assert_passage + GROUNDS exist to express. Belief-as-node-type already carries
# the belief itself. Aesthetic is likewise relational, or already covered by
# Anti/Value/Belief.
#
# "Tension" is kept despite also having zero instances: tensions are modelled as
# TENSIONS_WITH edges (13 of them) and find_tensions() reads the edges, so the
# node type is harmless dead schema rather than a miscategorisation risk.
NODE_TYPES = {"Project", "Concept", "Value", "Role", "Goal", "Tension",
              "Belief", "Person", "Anti", "Passage", "Text"}

# Edge types whose direction carries no meaning. "A OPPOSES B" and "B OPPOSES A"
# say the same thing, likewise TENSIONS_WITH -- but both are stored in a DiGraph,
# so every assertion had to pick an arbitrary direction, and the graph duly
# picked inconsistently. Measured 2026-08-13: of 20 OPPOSES edges, 12 ran
# Anti->Value ("attacker to victim"), 5 ran Concept->Anti ("defender to enemy"),
# and one pair -- epistemic offense / sycophancy -- was stored BOTH ways, which
# is what an arbitrary convention looks like when it finally contradicts itself.
#
# The cost was real: get_neighbors("reasoning integrity", direction="out") on
# OPPOSES returned NOTHING, because all four of its opposition edges point at
# it. The most connected node in the graph could not say what it stood against.
#
# Rather than re-author 20 edges, get_neighbors now ignores direction for these
# types. Direction was never carrying information here, only noise.
SYMMETRIC_EDGE_TYPES = {"OPPOSES", "TENSIONS_WITH"}

# Valid edge types
EDGE_TYPES = {
    "EXPRESSES",
    "SUPPORTS",
    "APPLIES_TO",
    "TENSIONS_WITH",
    "REQUIRES",
    "OPPOSES",
    "INFORMS",
    "EMBODIES",
    "PRODUCES",
    "ASSERTED_BY",
    "GROUNDS",       # Passage → node or edge claim it supports
    "SOURCED_FROM",  # Passage → Text it came from
    "AUTHORED_BY",   # Text → Person who wrote it
}


def stored_direction_word(d: str) -> str:
    """Map the stored label onto the direction argument callers pass."""
    return "out" if d == "outgoing" else "in"


class CalliopeGraph:
    def __init__(self, path: str = GRAPH_PATH):
        self.path = path
        self.G = nx.DiGraph()
        self._load()

    # -- Persistence ----------------------------------------------------------

    def _load(self):
        self.retired = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            # Deletions are moved here rather than destroyed. A graph that
            # cannot show what was removed cannot answer "how have I changed".
            self.retired = data.get("retired", [])
            for node in data.get("nodes", []):
                self.G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
            for edge in data.get("edges", []):
                self.G.add_edge(edge["source"], edge["target"], type=edge["type"],
                                **{k: v for k, v in edge.items() if k not in ("source", "target", "type")})

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "nodes": [{"id": n, **self.G.nodes[n]} for n in self.G.nodes],
            "edges": [{"source": u, "target": v, **self.G.edges[u, v]} for u, v in self.G.edges],
            "retired": getattr(self, "retired", []),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    # -- Queries --------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[dict]:
        """Return a node and all its edges, including weights and grounding."""
        if node_id not in self.G:
            matches = [n for n in self.G.nodes if n.lower() == node_id.lower()]
            if not matches:
                return None
            node_id = matches[0]

        attrs = dict(self.G.nodes[node_id])
        outgoing = [
            {
                "target": v,
                "type": self.G.edges[node_id, v]["type"],
                "weight": self.G.edges[node_id, v].get("weight", None),
                "grounding": self.G.edges[node_id, v].get("grounding", None),
            }
            for v in self.G.successors(node_id)
        ]
        incoming = [
            {
                "source": u,
                "type": self.G.edges[u, node_id]["type"],
                "weight": self.G.edges[u, node_id].get("weight", None),
                "grounding": self.G.edges[u, node_id].get("grounding", None),
            }
            for u in self.G.predecessors(node_id)
        ]
        return {
            "id": node_id,
            **attrs,
            "outgoing_edges": outgoing,
            "incoming_edges": incoming,
        }

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None, direction: str = "both") -> list:
        """Get neighbors of a node, optionally filtered by edge type and direction."""
        if node_id not in self.G:
            return []

        results = []
        seen = set()

        def take(other, edge, stored_direction):
            et = edge.get("type")
            if edge_type is not None and et != edge_type:
                return
            symmetric = et in SYMMETRIC_EDGE_TYPES
            # A symmetric edge is returned whichever way the caller asked,
            # because its stored direction is an artefact of whoever asserted
            # it rather than a claim about the world.
            if not symmetric and direction not in (stored_direction_word(stored_direction), "both"):
                return
            key = (other, et, stored_direction)
            if key in seen:
                return
            seen.add(key)
            results.append({
                "node": other,
                "edge_type": et,
                "weight": edge.get("weight", None),
                "grounding": edge.get("grounding", None),
                "direction": "symmetric" if symmetric else stored_direction,
                "node_attrs": dict(self.G.nodes[other]),
            })

        for v in self.G.successors(node_id):
            take(v, self.G.edges[node_id, v], "outgoing")
        for u in self.G.predecessors(node_id):
            take(u, self.G.edges[u, node_id], "incoming")
        return results

    def find_path(self, source: str, target: str) -> Optional[list]:
        """Find shortest path between two nodes, returning nodes, edge types, weights, and grounding."""
        try:
            path = nx.shortest_path(self.G.to_undirected(), source, target)
            result = []
            for i, node in enumerate(path):
                entry = {"node": node, "attrs": dict(self.G.nodes[node])}
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    if self.G.has_edge(node, next_node):
                        edge = self.G.edges[node, next_node]
                        entry["edge_to_next"] = edge.get("type")
                        entry["edge_weight"] = edge.get("weight", None)
                        entry["edge_grounding"] = edge.get("grounding", None)
                    else:
                        edge = self.G.edges[next_node, node]
                        entry["edge_to_next"] = edge.get("type") + " (reversed)"
                        entry["edge_weight"] = edge.get("weight", None)
                        entry["edge_grounding"] = edge.get("grounding", None)
                result.append(entry)
            return result
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_by_type(self, node_type: str) -> list:
        """Return all nodes of a given type."""
        return [
            {"id": n, **self.G.nodes[n]}
            for n in self.G.nodes
            if self.G.nodes[n].get("type") == node_type
        ]

    def find_tensions(self) -> list:
        """Return all TENSIONS_WITH edges, including weights and grounding, sorted by weight descending."""
        tensions = [
            {
                "node_a": u,
                "node_b": v,
                "weight": self.G.edges[u, v].get("weight", None),
                "grounding": self.G.edges[u, v].get("grounding", None),
                "attrs_a": dict(self.G.nodes[u]),
                "attrs_b": dict(self.G.nodes[v]),
            }
            for u, v in self.G.edges
            if self.G.edges[u, v].get("type") == "TENSIONS_WITH"
        ]
        return sorted(tensions, key=lambda t: t.get("weight") or 0.5, reverse=True)

    def find_values(self) -> list:
        """Return all Value nodes, sorted by weight descending so bedrock values come first."""
        nodes = self.find_by_type("Value")
        return sorted(nodes, key=lambda n: n.get("weight") or 0.5, reverse=True)

    def find_goals(self) -> list:
        """Return all Goal nodes."""
        return self.find_by_type("Goal")

    def search(self, query: str, top_k: int = 8, node_type: str = None) -> dict:
        """
        Search all nodes by MEANING, not by characters.

        v1 of this method was a substring match (`if query_lower in searchable`)
        whose docstring called itself "fuzzy search". It was not fuzzy. It found
        "reasoning integrity" for the query "honesty" only by accident, because
        the word "honestly" happened to sit in that node's description, and it
        found nothing at all for "intellectual courage" or "am I bullshitting
        myself". See graph_search.py for the full post-mortem.

        This delegates to the same Nomic embedding space the corpus has been
        using all along, so the two halves of Calliope now retrieve the same
        way. If LM Studio is unreachable it falls back to the old substring
        behaviour and SAYS SO via mode="lexical" -- degraded, but not silently.

        Returns {"mode": "semantic"|"lexical", "query": str, "results": [...]},
        where each result carries a `similarity` score and whether it matched
        by "name" or by "meaning".
        """
        # Imported inside the method so that `import graph` never depends on
        # the embedding stack being importable, and so there is no import cycle
        # with graph_search's own __main__ block.
        import graph_search
        return graph_search.search(self, query, top_k=top_k, node_type=node_type)

    def get_full_graph(self) -> dict:
        """Return the entire graph as a summary dict."""
        return {
            "node_count": self.G.number_of_nodes(),
            "edge_count": self.G.number_of_edges(),
            "nodes": [{"id": n, **self.G.nodes[n]} for n in self.G.nodes],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "type": self.G.edges[u, v].get("type"),
                    "weight": self.G.edges[u, v].get("weight", None),
                    "grounding": self.G.edges[u, v].get("grounding", None),
                }
                for u, v in self.G.edges
            ],
        }

    # -- Mutations ------------------------------------------------------------

    def _canonical_id(self, node_id: str):
        """
        Return the EXISTING node id that `node_id` refers to, ignoring case and
        whitespace, or None if there is no such node.

        get_node has always done this lookup leniently; assert_node did not.
        The asymmetry meant Calliope could READ "Reasoning Integrity" and find
        the node, but ASSERTING "Reasoning Integrity" silently created a second
        node beside "reasoning integrity" -- because node_id is the primary key
        and add_node does no normalisation. Graph integrity therefore depended
        on a 12B model reproducing 44 multi-word identifiers character-perfect
        from memory, across sessions. This closes that gap.
        """
        key = " ".join(node_id.split()).lower()
        for n in self.G.nodes:
            if " ".join(n.split()).lower() == key:
                return n
        return None

    def _resolve_provenance(self, provenance: str, refs: list) -> str:
        """
        Settle an assertion's provenance, computing `derived` where it applies.

        The rule: an inference grounded ONLY in other inferences is derived.
        An inference grounded in a corpus chunk is still `inferred`, because a
        chunk is primary evidence -- something Sean actually read -- not another
        of Calliope's conclusions.

        A reference that names nothing in the graph is treated as external
        evidence (a corpus chunk id). A parent whose own provenance is unknown
        -- which is every node predating this layer -- does NOT make its child
        derived; unknown provenance stays unknown rather than being promoted
        into a chain that may not exist.

        `stated` and `confirmed` are never downgraded. They are claims about who
        said a thing, not about what supports it.
        """
        p = (provenance or DEFAULT_PROVENANCE).strip().lower()
        if p not in PROVENANCE_LEVELS:
            p = DEFAULT_PROVENANCE
        if p != "inferred" or not refs:
            return p

        internal = [r for r in refs if r in self.G]
        if len(internal) < len(refs):
            return "inferred"          # at least one ref is primary evidence
        parents = [self.G.nodes[r].get("provenance") for r in internal]
        if parents and all(x in ("inferred", "derived") for x in parents):
            return "derived"
        return "inferred"

    def find_orphans(self, max_degree: int = 0) -> list:
        """
        Nodes with max_degree edges or fewer -- by default, none at all.

        An unconnected node is a note, not an ontology entry: it can be
        retrieved by search but never reasoned THROUGH, and find_path can never
        reach it. Seven of the first 54 nodes were orphans, including Sean's
        wife, so this is a live condition rather than a hypothetical.

        Exposed as a tool so Calliope can go looking for unfinished work
        instead of waiting to stumble over it.
        """
        out = []
        for n in self.G.nodes:
            deg = self.G.degree(n)
            if deg <= max_degree:
                out.append({"id": n, "degree": deg, **dict(self.G.nodes[n])})
        return sorted(out, key=lambda d: (d["degree"], d.get("type") or "", d["id"]))

    def find_neglected(self, limit: int = 10) -> list:
        """
        Nodes the conversation keeps passing over.

        find_orphans catches STRUCTURAL neglect -- nothing connects here.
        This catches ATTENTIONAL neglect -- nothing has touched this in a long
        time, or ever. The two rarely overlap: a node can sit on three edges and
        still never come up.

        Why it exists. Measured on this graph: Value nodes average 6.3 edges,
        Person nodes 2.5. That gap is not a fact about Sean, it is a fact about
        retrieval. `find_values` and `find_tensions` fire on EVERY turn, so
        every Value and every tension participant is permanently in view, while
        a Concept or a Person is seen only if someone goes looking. Hub nodes
        also carry richer descriptions -- 39 words against 15 for the sparse
        ones -- so they win semantic search too, and get elaborated further.
        Attention compounds into structure, and the graph slowly becomes a map
        of what was already central rather than of what is true.

        Sorted so the never-touched and least-connected surface first.
        """
        out = []
        for n in self.G.nodes:
            a = self.G.nodes[n]
            out.append({
                "id": n,
                "type": a.get("type"),
                "degree": self.G.degree(n),
                "last_updated": a.get("last_updated") or "never recorded",
                "description": (a.get("description") or "")[:140],
            })
        out.sort(key=lambda d: (d["last_updated"] != "never recorded",
                                d["last_updated"], d["degree"]))
        return out[:limit]

    def assert_node(self, node_id: str, node_type: str, description: str = "",
                    weight: float = DEFAULT_WEIGHT, provenance: str = DEFAULT_PROVENANCE,
                    grounded_in="", **attrs) -> dict:
        """
        Add or update a node. Weight is credence: 0.0 (tentative) to 1.0 (bedrock).

        Updating an existing node MERGES rather than replaces. networkx's
        add_node overwrites only the keys you pass, but v1 always passed
        description and weight, so re-asserting a node with a blank description
        wiped the existing one. Now an empty description leaves the old text
        alone.

        Three things happen here beyond the write itself, all of them advisory:
          * the id is resolved case/whitespace-insensitively against existing
            nodes, so near-miss spellings update rather than duplicate;
          * a genuinely new node is checked against the embedding index for
            near-duplicates and any match is REPORTED, never blocked;
          * a new node that ends up with no edges is flagged as an orphan.
        None of these can fail the assertion. A dead embedding server must not
        make the graph unwritable.
        """
        if node_type not in NODE_TYPES:
            return {"error": f"Unknown node type '{node_type}'. Valid types: {sorted(NODE_TYPES)}"}
        weight = max(0.0, min(1.0, float(weight)))

        canonical = self._canonical_id(node_id)
        resolved_from = node_id if (canonical and canonical != node_id) else None
        if canonical:
            node_id = canonical

        is_new = node_id not in self.G

        # Duplicate check BEFORE the write, so the index holds only pre-existing
        # nodes and the candidate cannot match itself.
        similar = []
        if is_new:
            try:
                import graph_search
                similar = graph_search.find_similar_node(
                    self, node_id, node_type, description)
            except Exception:
                similar = []

        existing = dict(self.G.nodes[node_id]) if node_id in self.G else {}
        if not description:
            description = existing.get("description", "")

        # -- the temporal trail ------------------------------------------------
        # Before this existed, assert_node merged over the top and the previous
        # description was simply gone. Change was not merely unrecorded, it was
        # destroyed on write -- in a graph whose own contents include "belief
        # residue" and "epistemic grief".
        history = list(existing.get("history", []))
        if is_new:
            first_asserted = _today()
        else:
            first_asserted = existing.get("first_asserted", PRE_TEMPORAL)
            old_w = existing.get("weight")
            if old_w is not None and abs(float(old_w) - weight) > 1e-9:
                history.append({"date": _today(),
                                "change": f"weight {old_w} -> {weight}"})
            old_d = existing.get("description", "")
            if old_d and description and old_d != description:
                history.append({"date": _today(), "change": "description rewritten",
                                "was": old_d})
            old_t = existing.get("type")
            if old_t and old_t != node_type:
                history.append({"date": _today(),
                                "change": f"type {old_t} -> {node_type}"})

        refs = _refs(grounded_in)
        resolved = self._resolve_provenance(provenance, refs)
        old_p = existing.get("provenance")
        if old_p and old_p != resolved:
            history.append({"date": _today(),
                            "change": f"provenance {old_p} -> {resolved}"})

        self.G.add_node(node_id, type=node_type, description=description,
                        weight=weight, provenance=resolved, grounded_in=refs,
                        first_asserted=first_asserted,
                        last_updated=_today(), history=history, **attrs)
        self.save()

        result = {"status": "ok", "node": self.get_node(node_id),
                  "provenance": resolved}
        if resolved != (provenance or DEFAULT_PROVENANCE).strip().lower():
            result["provenance_note"] = (
                f"Recorded as '{resolved}': every id in grounded_in is itself an "
                f"inference, so this is one step further from the evidence.")
        unknown = [r for r in refs if r not in self.G]
        if unknown:
            result["grounded_in_external"] = unknown
            result["grounding_note"] = (
                f"{len(unknown)} reference(s) name nothing in the graph and were "
                f"recorded as external evidence (corpus chunk ids). If one was meant "
                f"to be a node, check the spelling -- it will not resolve later.")

        if resolved_from:
            result["resolved_to_existing_id"] = node_id
            result["note"] = (
                f"'{resolved_from}' resolved to the existing node '{node_id}' "
                f"and updated it rather than creating a second node.")

        if similar:
            result["similar_existing"] = similar
            result["warning"] = (
                "This node closely resembles one that already exists. If it says the "
                "same thing, consider re-asserting the existing node with a richer "
                "description and connecting to that instead -- two nodes for one idea "
                "split its edges and weaken both. If it is genuinely distinct, keep it "
                "and add an edge saying how the two relate.")

        if is_new and self.G.degree(node_id) == 0:
            result["orphan"] = True
            result["next_step"] = (
                f"'{node_id}' has no edges yet, so nothing can reason through it. "
                f"Add at least one assert_edge connecting it to something already in "
                f"the graph before you finish this turn.")

        return result

    def assert_edge(self, source: str, target: str, edge_type: str,
                    weight: float = DEFAULT_WEIGHT, grounding: str = "",
                    provenance: str = DEFAULT_PROVENANCE, grounded_in="",
                    **attrs) -> dict:
        """Add or update an edge. Weight is credence; grounding is the justification for this claim."""
        if edge_type not in EDGE_TYPES:
            return {"error": f"Unknown edge type '{edge_type}'. Valid types: {sorted(EDGE_TYPES)}"}
        if source not in self.G:
            return {"error": f"Source node '{source}' does not exist."}
        if target not in self.G:
            return {"error": f"Target node '{target}' does not exist."}
        weight = max(0.0, min(1.0, float(weight)))

        existing = dict(self.G.edges[source, target]) if self.G.has_edge(source, target) else {}
        history = list(existing.get("history", []))
        if existing:
            first_asserted = existing.get("first_asserted", PRE_TEMPORAL)
            old_w = existing.get("weight")
            if old_w is not None and abs(float(old_w) - weight) > 1e-9:
                history.append({"date": _today(),
                                "change": f"weight {old_w} -> {weight}"})
            old_t = existing.get("type")
            if old_t and old_t != edge_type:
                history.append({"date": _today(),
                                "change": f"type {old_t} -> {edge_type}"})
            old_g = existing.get("grounding") or ""
            if old_g and grounding and old_g != grounding:
                history.append({"date": _today(), "change": "grounding rewritten",
                                "was": old_g})
        else:
            first_asserted = _today()

        if not grounding:
            grounding = existing.get("grounding", "")

        refs = _refs(grounded_in)
        resolved = self._resolve_provenance(provenance, refs)
        old_p = existing.get("provenance")
        if old_p and old_p != resolved:
            history.append({"date": _today(),
                            "change": f"provenance {old_p} -> {resolved}"})

        self.G.add_edge(source, target, type=edge_type, weight=weight,
                        grounding=grounding, provenance=resolved, grounded_in=refs,
                        first_asserted=first_asserted,
                        last_updated=_today(), history=history, **attrs)
        self.save()
        return {
            "status": "ok",
            "edge": {
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": weight,
                "grounding": grounding,
                "provenance": resolved,
                "grounded_in": refs,
                "first_asserted": first_asserted,
            }
        }

    def assert_passage(self, passage_id: str, text: str, author: str, title: str,
                       grounds=None, weight: float = 0.8,
                       source: str = "", chunk_id: str = "") -> dict:
        """
        Reify a corpus passage as a node, and ground claims in it.

        REWRITTEN 2026-08-14. The previous version minted ids that could not
        join anything and bypassed every safeguard on the class:

          * Person id was `author.lower().replace(" ", "_")` -> "slavoj_zizek",
            while the graph already held "Slavoj Zizek". First use would have
            created a duplicate person, silently.
          * Text id came from the TITLE -> "the_sublime_object_of_ideology",
            while the corpus keys on the file stem -> "sublime_object_zizek".
            No join was possible in either direction.
          * Passage id was md5(text[:100]); the corpus chunk id is
            md5(stem|index|text[:200]). Same passage, two unrelated hashes.
          * It called self.G.add_node() directly rather than self.assert_node(),
            so it skipped canonical-id resolution, the near-duplicate check and
            all temporal stamping.

        Now: people and texts are named the way a human would name them, so
        canonicalisation resolves them against what is already there; the corpus
        `source` stem rides along as an attribute so the two stores can be
        joined; and the passage is keyed by its real `chunk_id` when one is
        given, which is what makes a graph claim traceable to the exact chunk
        that supports it.

        Everything goes through assert_node, so all of it inherits duplicate
        detection, id resolution and history.
        """
        import hashlib
        weight = max(0.0, min(1.0, float(weight)))
        grounds = _refs(grounds)

        # Passage: prefer the corpus's own id. That single choice is what turns
        # "a passage like this one" into "this passage, chunk 41f9c2...".
        if not passage_id:
            passage_id = chunk_id or ("passage_" + hashlib.md5(
                text[:100].encode()).hexdigest()[:8])

        self.assert_node(
            passage_id, "Passage", text[:500], weight=weight,
            provenance="stated", full_text=text, author=author, title=title,
            source=source or "", chunk_id=chunk_id or "",
        )

        # Text and Person keep human-readable ids so _canonical_id can match the
        # nodes already in the graph. `source` is the join key to the corpus.
        text_node = title.strip()
        self.assert_node(text_node, "Text", title.strip(), weight=1.0,
                         provenance="stated", author=author, source=source or "")

        person_node = author.strip()
        self.assert_node(person_node, "Person", author.strip(), weight=1.0,
                         provenance="stated")

        self.assert_edge(passage_id, text_node, "SOURCED_FROM", 1.0,
                         "direct source", provenance="stated")
        self.assert_edge(text_node, person_node, "AUTHORED_BY", 1.0,
                         "authorship", provenance="stated")

        # The GROUNDS edge is the only judgement here: the passage existing is a
        # fact, but "this passage supports that claim" is Calliope's reading.
        grounding_results = []
        for target_node in grounds:
            resolved = self._canonical_id(target_node)
            if resolved:
                self.assert_edge(
                    passage_id, resolved, "GROUNDS", weight,
                    f"Passage from {author}, '{title}' supports this claim.",
                    provenance="inferred", grounded_in=[passage_id],
                )
                grounding_results.append({"target": resolved, "status": "ok"})
            else:
                grounding_results.append(
                    {"target": target_node,
                     "status": "no such node -- skipped; check the exact node id"})

        self.save()
        return {
            "status": "ok",
            "passage_id": passage_id,
            "text_node": text_node,
            "person_node": person_node,
            "joined_to_corpus_by": source or chunk_id or "(nothing -- pass source or chunk_id)",
            "grounds": grounding_results,
        }

    # -- Maintenance ----------------------------------------------------------

    def audit(self) -> dict:
        """
        Report structural gaps in the graph.

        This exists because of a specific silent failure: Calliope's system
        prompt devotes several paragraphs to reading weights ("a value with
        weight 0.9 is bedrock -- hold Sean to it hard; 0.4 is tentative --
        probe it") and insists every edge carry a grounding ("a claim without
        a grounding is not a claim, it's a guess").

        In the seed graph, 47 of 48 nodes and 95 of 95 edges had NO weight, and
        no edge had any grounding at all. Every one of those instructions was
        describing machinery that wasn't connected to anything. Sorting by
        weight fell back to a hardcoded 0.5 for everything, so "sorted by
        weight so bedrock values come first" produced arbitrary order.

        Nothing here can be auto-fixed: weights are Sean's judgments about his
        own commitments, and inventing them would be exactly the kind of
        confident fabrication the whole design is against.
        """
        unweighted_nodes = [n for n in self.G.nodes
                            if self.G.nodes[n].get("weight") is None]
        unweighted_edges = [(u, v) for u, v in self.G.edges
                            if self.G.edges[u, v].get("weight") is None]
        ungrounded_edges = [(u, v) for u, v in self.G.edges
                            if not self.G.edges[u, v].get("grounding")]
        undescribed = [n for n in self.G.nodes
                       if not self.G.nodes[n].get("description")]
        unknown_types = [n for n in self.G.nodes
                         if self.G.nodes[n].get("type") not in NODE_TYPES]
        orphans = [n for n in self.G.nodes if self.G.degree(n) == 0]

        return {
            "node_count": self.G.number_of_nodes(),
            "edge_count": self.G.number_of_edges(),
            "unweighted_nodes": unweighted_nodes,
            "unweighted_edges": unweighted_edges,
            "ungrounded_edges": ungrounded_edges,
            "undescribed_nodes": undescribed,
            "unknown_node_types": unknown_types,
            "orphan_nodes": orphans,
        }

    def set_weight(self, node_id: str, weight: float) -> dict:
        """Set a node's credence without touching anything else about it."""
        if node_id not in self.G:
            return {"error": f"Node '{node_id}' not found."}
        self.G.nodes[node_id]["weight"] = max(0.0, min(1.0, float(weight)))
        self.save()
        return {"status": "ok", "node_id": node_id,
                "weight": self.G.nodes[node_id]["weight"]}

    def remove_node(self, node_id: str, reason: str = "") -> dict:
        """
        Retire a node and its edges. NOT a hard delete.

        The node leaves the live graph -- traversal, search and embedding all
        stop seeing it immediately -- but its final state is preserved in the
        `retired` list with a date. Dropping a belief is exactly the kind of
        change this graph exists to remember; destroying the evidence of it
        would be the opposite of memory.
        """
        if node_id not in self.G:
            return {"error": f"Node '{node_id}' not found."}
        attrs = dict(self.G.nodes[node_id])
        edges = [
            {"source": u, "target": v, **self.G.edges[u, v]}
            for u, v in list(self.G.in_edges(node_id)) + list(self.G.out_edges(node_id))
        ]
        self.retired.append({
            "kind": "node", "id": node_id, "retired_at": _today(),
            "reason": reason, "attrs": attrs, "edges": edges,
        })
        self.G.remove_node(node_id)
        self.save()
        return {"status": "ok", "retired": node_id,
                "note": f"Retired, not deleted. {len(edges)} edge(s) retired with it. "
                        f"Recoverable from the graph's retired list."}

    def remove_edge(self, source: str, target: str, reason: str = "") -> dict:
        """Retire an edge. NOT a hard delete -- see remove_node."""
        if not self.G.has_edge(source, target):
            return {"error": f"Edge '{source}' -> '{target}' not found."}
        attrs = dict(self.G.edges[source, target])
        self.retired.append({
            "kind": "edge", "source": source, "target": target,
            "retired_at": _today(), "reason": reason, "attrs": attrs,
        })
        self.G.remove_edge(source, target)
        self.save()
        return {"status": "ok", "retired": f"{source} -> {target}",
                "note": "Retired, not deleted. Recoverable from the retired list."}

    # -- Memory ---------------------------------------------------------------

    def what_changed(self, since: str = "", limit: int = 40) -> dict:
        """
        What has moved in this graph, most recent first.

        This is the tool that lets Calliope answer "how have I changed?" rather
        than only "what do I think?". It reports three kinds of movement:
        things newly asserted, things whose weight or wording shifted, and
        things retired.

        `since` is an ISO date (YYYY-MM-DD). Anything carrying the PRE_TEMPORAL
        marker is reported separately and never dated, because its real date is
        genuinely unknown -- inventing one would corrupt the only record of
        change the graph has.
        """
        events = []

        def consider(kind, label, attrs):
            first = attrs.get("first_asserted")
            if first and first != PRE_TEMPORAL and (not since or first >= since):
                events.append({"date": first, "event": "asserted",
                               "kind": kind, "what": label})
            for h in attrs.get("history", []):
                d = h.get("date", "")
                if not since or d >= since:
                    e = {"date": d, "event": h.get("change", "changed"),
                         "kind": kind, "what": label}
                    if h.get("was"):
                        e["was"] = h["was"][:200]
                    events.append(e)

        for n in self.G.nodes:
            consider("node", n, self.G.nodes[n])
        for u, v in self.G.edges:
            a = self.G.edges[u, v]
            consider("edge", f"{u} --[{a.get('type')}]--> {v}", a)
        for r in getattr(self, "retired", []):
            d = r.get("retired_at", "")
            if not since or d >= since:
                label = (r["id"] if r["kind"] == "node"
                         else f"{r['source']} --> {r['target']}")
                ev = {"date": d, "event": "retired", "kind": r["kind"], "what": label}
                if r.get("reason"):
                    ev["reason"] = r["reason"]
                events.append(ev)

        events.sort(key=lambda e: e.get("date", ""), reverse=True)
        undated = sum(1 for n in self.G.nodes
                      if self.G.nodes[n].get("first_asserted", PRE_TEMPORAL) == PRE_TEMPORAL
                      or "first_asserted" not in self.G.nodes[n])

        return {
            "since": since or "all time",
            "events": events[:limit],
            "total_events": len(events),
            "undated_nodes": undated,
            "note": (f"{undated} node(s) predate the temporal layer and have no known "
                     f"assertion date. They are not missing from the graph, only from "
                     f"its history -- do not describe them as recent or as unchanged."
                     if undated else None),
        }


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    g = CalliopeGraph()

    if "--audit" in _sys.argv:
        a = g.audit()
        print(f"\nGraph audit: {a['node_count']} nodes, {a['edge_count']} edges\n")

        def show(label, items, limit=12):
            if not items:
                print(f"  OK    {label}: none")
                return
            print(f"  ISSUE {label}: {len(items)}")
            for x in items[:limit]:
                print(f"          - {x}")
            if len(items) > limit:
                print(f"          ... and {len(items) - limit} more")

        show("nodes with no weight", a["unweighted_nodes"])
        show("edges with no weight", a["unweighted_edges"])
        show("edges with no grounding", a["ungrounded_edges"])
        show("nodes with no description", a["undescribed_nodes"])
        show("nodes with an unrecognised type", a["unknown_node_types"])
        show("orphan nodes (no edges)", a["orphan_nodes"])

        print("\nWeights and groundings cannot be auto-filled -- they are your")
        print("judgments, not facts to be derived. Set them with:")
        print("    python3 -c \"from graph import CalliopeGraph as G; "
              "g=G(); g.set_weight('reasoning integrity', 0.9)\"\n")
        _sys.exit(0)

    s = g.get_full_graph()
    print(f"Graph: {s['node_count']} nodes, {s['edge_count']} edges")
    print("Run with --audit to see structural gaps.")

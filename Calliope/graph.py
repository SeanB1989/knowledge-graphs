"""
Calliope Graph Engine
A live, traversable personal ontology and epistemology graph.
"""

import json
import os
from typing import Optional
import networkx as nx


GRAPH_PATH = os.path.join(os.path.dirname(__file__), "data", "graph.json")

# Valid node types
NODE_TYPES = {"Project", "Concept", "Value", "Role", "Goal", "Tension", "Aesthetic", "Epistemic", "Person", "Anti"}

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
}


class CalliopeGraph:
    def __init__(self, path: str = GRAPH_PATH):
        self.path = path
        self.G = nx.DiGraph()
        self._load()

    # -- Persistence ----------------------------------------------------------

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
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
        if direction in ("out", "both"):
            for v in self.G.successors(node_id):
                edge = self.G.edges[node_id, v]
                if edge_type is None or edge.get("type") == edge_type:
                    results.append({
                        "node": v,
                        "edge_type": edge.get("type"),
                        "weight": edge.get("weight", None),
                        "grounding": edge.get("grounding", None),
                        "direction": "outgoing",
                        "node_attrs": dict(self.G.nodes[v]),
                    })
        if direction in ("in", "both"):
            for u in self.G.predecessors(node_id):
                edge = self.G.edges[u, node_id]
                if edge_type is None or edge.get("type") == edge_type:
                    results.append({
                        "node": u,
                        "edge_type": edge.get("type"),
                        "weight": edge.get("weight", None),
                        "grounding": edge.get("grounding", None),
                        "direction": "incoming",
                        "node_attrs": dict(self.G.nodes[u]),
                    })
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

    def search(self, query: str) -> list:
        """Fuzzy search across all node IDs and attributes."""
        query_lower = query.lower()
        results = []
        for node in self.G.nodes:
            attrs = self.G.nodes[node]
            searchable = node.lower() + " " + " ".join(str(v).lower() for v in attrs.values())
            if query_lower in searchable:
                results.append({"id": node, **attrs})
        return results

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

    def assert_node(self, node_id: str, node_type: str, description: str = "",
                    weight: float = 1.0, **attrs) -> dict:
        """Add or update a node. Weight is credence: 0.0 (tentative) to 1.0 (bedrock)."""
        if node_type not in NODE_TYPES:
            return {"error": f"Unknown node type '{node_type}'. Valid types: {sorted(NODE_TYPES)}"}
        weight = max(0.0, min(1.0, float(weight)))
        self.G.add_node(node_id, type=node_type, description=description, weight=weight, **attrs)
        self.save()
        return {"status": "ok", "node": self.get_node(node_id)}

    def assert_edge(self, source: str, target: str, edge_type: str,
                    weight: float = 1.0, grounding: str = "", **attrs) -> dict:
        """Add or update an edge. Weight is credence; grounding is the justification for this claim."""
        if edge_type not in EDGE_TYPES:
            return {"error": f"Unknown edge type '{edge_type}'. Valid types: {sorted(EDGE_TYPES)}"}
        if source not in self.G:
            return {"error": f"Source node '{source}' does not exist."}
        if target not in self.G:
            return {"error": f"Target node '{target}' does not exist."}
        weight = max(0.0, min(1.0, float(weight)))
        self.G.add_edge(source, target, type=edge_type, weight=weight, grounding=grounding, **attrs)
        self.save()
        return {
            "status": "ok",
            "edge": {
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": weight,
                "grounding": grounding,
            }
        }

    def remove_node(self, node_id: str) -> dict:
        """Remove a node and all its edges."""
        if node_id not in self.G:
            return {"error": f"Node '{node_id}' not found."}
        self.G.remove_node(node_id)
        self.save()
        return {"status": "ok", "removed": node_id}

    def remove_edge(self, source: str, target: str) -> dict:
        """Remove an edge."""
        if not self.G.has_edge(source, target):
            return {"error": f"Edge '{source}' -> '{target}' not found."}
        self.G.remove_edge(source, target)
        self.save()
        return {"status": "ok", "removed": f"{source} -> {target}"}

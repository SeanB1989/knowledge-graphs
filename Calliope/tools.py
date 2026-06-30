"""
Calliope Tool Layer
Defines the tools Calliope can call to navigate the personal ontology graph.
Now using OpenAI native function calling format -- no more XML parsing.
"""

import json
from graph import CalliopeGraph


# -- Tool Definitions in OpenAI function calling format ----------------------
# These are passed directly to the API as the `tools` parameter.
# The model receives them as structured schemas, not as text in the prompt.

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_node",
            "description": "Fetch a specific node by ID, returning its type, description, weight, and all connected edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The exact node identifier (e.g. 'reasoning integrity', 'Calliope')"
                    }
                },
                "required": ["node_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": "Get all nodes connected to a given node, optionally filtered by edge type or direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "The node to expand"},
                    "edge_type": {"type": "string", "description": "Filter by edge type e.g. TENSIONS_WITH, SUPPORTS, INFORMS"},
                    "direction": {"type": "string", "enum": ["in", "out", "both"], "description": "in, out, or both (default: both)"}
                },
                "required": ["node_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_path",
            "description": "Find how two nodes are connected -- returns the path of nodes and edge types between them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Starting node ID"},
                    "target": {"type": "string", "description": "Destination node ID"}
                },
                "required": ["source", "target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_tensions",
            "description": "Return all TENSIONS_WITH relationships in the graph -- the live conflicts and dialectics. Includes weights so you can see which tensions are acute vs. mild.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_values",
            "description": "Return all Value nodes -- the moral and aesthetic commitments encoded in the graph. Sorted by weight so bedrock values come first.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_goals",
            "description": "Return all Goal nodes -- the desired outcomes and directions of travel.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search across all node IDs and descriptions by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword to search for"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_graph",
            "description": "Return a complete summary of the graph -- all nodes and edges. Use sparingly; prefer targeted queries.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assert_node",
            "description": (
                "Add a new node to the graph or update an existing one. "
                "Set weight to reflect certainty (0.0=tentative, 1.0=bedrock). Default 0.7 for new inferences. "
                "Values are moral commitments you aspire toward -- not fears, anxieties, or psychological patterns. "
                "When in doubt, use Concept."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Unique identifier for this node"},
                    "node_type": {
                        "type": "string",
                        "enum": ["Project", "Concept", "Value", "Role", "Goal", "Tension", "Aesthetic", "Epistemic", "Person", "Anti"],
                        "description": "Node type. Values=aspirational commitments. Concepts=forces/patterns. Anti=named enemies. Epistemic=belief stances."
                    },
                    "description": {"type": "string", "description": "What this node means"},
                    "weight": {"type": "number", "description": "Credence 0.0 to 1.0. Default 0.7 for new inferences."}
                },
                "required": ["node_id", "node_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assert_edge",
            "description": "Add a new edge between two existing nodes. Include weight and grounding -- every claim should be justified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source node ID"},
                    "target": {"type": "string", "description": "Target node ID"},
                    "edge_type": {
                        "type": "string",
                        "enum": ["EXPRESSES", "SUPPORTS", "APPLIES_TO", "TENSIONS_WITH", "REQUIRES", "OPPOSES", "INFORMS", "EMBODIES", "PRODUCES", "ASSERTED_BY"],
                        "description": "The type of relationship"
                    },
                    "weight": {"type": "number", "description": "Credence 0.0 to 1.0. How confident are you this relationship is real?"},
                    "grounding": {"type": "string", "description": "Why do you believe this relationship holds? Cite the source."}
                },
                "required": ["source", "target", "edge_type"]
            }
        }
    },
]


def tool_definitions_for_api() -> list:
    """Return tool definitions in OpenAI function calling format for the API."""
    return TOOL_DEFINITIONS


def tool_definitions_as_text() -> str:
    """Render tool definitions as plain text. Used for system prompt summaries."""
    lines = []
    for t in TOOL_DEFINITIONS:
        fn = t["function"]
        lines.append(f"- {fn['name']}: {fn['description']}")
    return "\n".join(lines)


# -- Tool Execution -----------------------------------------------------------

def execute_tool(graph: CalliopeGraph, tool_name: str, args: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        if tool_name == "get_node":
            result = graph.get_node(args["node_id"])
            if result is None:
                result = {"error": f"Node '{args['node_id']}' not found in graph."}

        elif tool_name == "get_neighbors":
            result = graph.get_neighbors(
                args["node_id"],
                edge_type=args.get("edge_type"),
                direction=args.get("direction", "both")
            )

        elif tool_name == "find_path":
            result = graph.find_path(args["source"], args["target"])
            if result is None:
                result = {"error": f"No path found between '{args['source']}' and '{args['target']}'."}

        elif tool_name == "find_tensions":
            result = graph.find_tensions()

        elif tool_name == "find_values":
            result = graph.find_values()

        elif tool_name == "find_goals":
            result = graph.find_goals()

        elif tool_name == "search":
            result = graph.search(args["query"])

        elif tool_name == "get_full_graph":
            result = graph.get_full_graph()

        elif tool_name == "assert_node":
            result = graph.assert_node(
                args["node_id"],
                args["node_type"],
                description=args.get("description", ""),
                weight=args.get("weight", 0.7)
            )

        elif tool_name == "assert_edge":
            result = graph.assert_edge(
                args["source"],
                args["target"],
                args["edge_type"],
                weight=args.get("weight", 0.7),
                grounding=args.get("grounding", "")
            )

        else:
            result = {"error": f"Unknown tool: '{tool_name}'"}

    except KeyError as e:
        result = {"error": f"Missing required parameter: {e}"}
    except Exception as e:
        result = {"error": str(e)}

    return json.dumps(result, indent=2)

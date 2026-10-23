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
                    "direction": {"type": "string", "description": "Must be exactly 'in', 'out', or 'both'. Default 'both'."}
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
            "name": "find_neglected",
            "description": (
                "Nodes the conversation keeps passing over -- least recently touched "
                "first, never-touched before that. Different from find_orphans: a node "
                "can sit on three edges and still never come up. Use this deliberately, "
                "because your own retrieval habits bias you: find_values and "
                "find_tensions run every turn, so Values and tensions are permanently in "
                "view while Concepts and People are seen only if you go looking, and the "
                "nodes with the richest descriptions also win semantic search. Left "
                "alone, that turns the graph into a map of what was already central "
                "rather than what is true. Go and look at the edges of it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer",
                              "description": "How many to return. Default 10."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "what_changed",
            "description": (
                "What has moved in the graph, most recent first -- things newly asserted, "
                "weights and wordings that shifted, and things retired. This is how you "
                "answer questions about how Sean has CHANGED, rather than only what he "
                "currently thinks. Use it when he asks what's different, when a belief "
                "seems to have moved since you last discussed it, or when you want to "
                "notice a drift he hasn't named. Nodes predating the temporal layer have "
                "no known date and are reported separately -- never describe those as "
                "recent, or as unchanged."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since": {
                        "type": "string",
                        "description": "Optional ISO date, YYYY-MM-DD. Omit for all recorded history."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_orphans",
            "description": (
                "Find nodes that are unconnected or barely connected -- ontology entries "
                "that exist but are not wired into anything. An orphan can be retrieved by "
                "search but nothing can reason THROUGH it and find_path can never reach it, "
                "so it contributes nothing to how you understand Sean. Use this to find "
                "unfinished work and propose edges, especially when a conversation touches "
                "something an orphan is obviously about. Returns each node with its degree."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_degree": {
                        "type": "integer",
                        "description": (
                            "Return nodes with this many edges or fewer. 0 (default) means "
                            "fully unconnected; 1 also catches nodes hanging off a single "
                            "thread."
                        )
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the graph by MEANING, not by keyword. Uses the same embedding "
                "model as search_corpus, so a query finds nodes that are conceptually "
                "related even when they share no words with it -- searching 'am I "
                "fooling myself' will surface 'reasoning integrity' and 'simulated "
                "knowledge'. Ask it the question you actually mean rather than guessing "
                "at node names. Returns {mode, query, results}, where each result has a "
                "similarity score and whether it matched by name or by meaning. If mode "
                "is 'lexical' the embedding server was unreachable and this degraded to "
                "a plain substring match -- say so rather than trusting the results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What you want to find. A phrase or question works better "
                            "than a single keyword."
                        ),
                    },
                    "node_type": {
                        "type": "string",
                        "description": (
                            "Optional. Restrict results to one category. Use this when "
                            "you want Sean's own commitments rather than his influences "
                            "-- an unfiltered conceptual query will often surface Person "
                            "nodes, because each thinker's description is a dense "
                            "summary of what they represent. One of: Project, Concept, "
                            "Value, Role, Goal, Tension, Belief, Person, Anti, Passage, Text."
                        ),
                    }
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
                        "description": (
                            "Node type -- must be exactly one of: Project, Concept, Value, "
                            "Role, Goal, Tension, Belief, Person, Anti. "
                            "Values=aspirational commitments. Concepts=forces/patterns. "
                            "Anti=named enemies. Belief=held convictions about how things are "
                            "(distinct from Value, which is aspirational). Where a belief comes "
                            "from is NOT a node type -- use assert_passage to ground it."
                        )
                    },
                    "description": {"type": "string", "description": "What this node means"},
                    "provenance": {
                        "type": "string",
                        "description": (
                            "Where this came from. Must be exactly one of: 'stated' "
                            "(Sean said it in this conversation or wrote it -- use this "
                            "only when you can point at his actual words), 'confirmed' "
                            "(you proposed it and he agreed), 'inferred' (you worked it "
                            "out -- THE DEFAULT, and the honest choice for most of what "
                            "you assert). Do not claim 'stated' for something you "
                            "concluded; that launders your inference into his voice. "
                            "'derived' is computed for you and should not be passed."
                        )
                    },
                    "grounded_in": {
                        "type": "string",
                        "description": (
                            "Comma-separated evidence for this assertion: corpus "
                            "chunk_id values returned by search_corpus, and/or ids of "
                            "graph nodes it rests on. This is a REFERENCE, not a "
                            "sentence -- it is what makes the claim traceable later. "
                            "If everything here is itself an inference, provenance is "
                            "automatically recorded as 'derived'."
                        )
                    },
                    "weight": {"type": "number", "description": "Credence 0.0 to 1.0. Default 0.7 for new inferences."}
                },
                "required": ["node_id", "node_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Hybrid search over Sean's personal corpus — the books, essays, and texts "
                "that have actually shaped his thinking. Combines meaning-based and "
                "keyword-based search, then reranks. Use it to find passages that ground "
                "a claim, illuminate a value, or provide evidence for an edge. "
                "Each result carries a `confidence` of strong, moderate or weak, and a "
                "ready-made `citation` string including page numbers — quote that verbatim "
                "rather than composing your own. If the result set is empty, or every hit "
                "is weak, say so plainly instead of citing it as if it were solid. "
                "Each result also carries a `chunk_id` and a `source`: pass the chunk_id "
                "as `grounded_in` when you assert something because of that passage, and "
                "pass both to assert_passage. That is what makes a claim traceable to its "
                "evidence rather than merely plausible. NOTE: `relevance` and `confidence` "
                "describe how well a passage matched YOUR QUERY. They are not credences "
                "about Sean, and must never be reused as an assertion's weight."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What you're looking for — a concept, claim, or question. "
                            "Full phrases work better than single words: the meaning-based "
                            "half of the search needs enough text to locate a position."
                        )
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of passages to return (default 5, max 10)",
                    },
                    "provenance": {
                        "type": "string",
                        "description": (
                            "Optional filter. Must be exactly 'sean', 'published', or "
                            "'dialogue'. 'sean' searches ONLY Sean's own writing; "
                            "'published' only the books; 'dialogue' only saved exchanges "
                            "with you -- which are useful for recalling what was already "
                            "worked through, but are NOT Sean speaking and must never be "
                            "cited as his position. Omit to search everything. "
                            "SEARCH 'sean' FIRST. His writing is about 2.5% of the corpus by "
                            "chunk count, so on any question about what HE thinks it will "
                            "lose to published authors on topical similarity alone. Look for "
                            "his own words before reaching for anyone else's, and prefer "
                            "grounding a claim of his in a passage he wrote."
                        )
                    },
                    "author": {
                        "type": "string",
                        "description": (
                            "Optional filter to one author's work, e.g. 'Mark Fisher'. "
                            "Must match exactly as stored."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assert_passage",
            "description": (
                "Reify a corpus passage as a first-class node in the graph. This is how a "
                "belief gets an epistemic source: the passage becomes a node, and GROUNDS "
                "edges record what it supports and why. "
                "Call this when search_corpus returns a passage worth preserving — "
                "one that genuinely grounds a claim, supports a value, or illuminates a tension. "
                "PREFER SEAN'S OWN WRITING. If a claim of his can be grounded in something he "
                "wrote, ground it there first; reach for a published author when his own work "
                "does not carry the point, or when the outside voice is doing work his cannot. "
                "Creates a Passage node, a Text node for the source work, a Person node for the author, "
                "and GROUNDS edges to any nodes this passage supports. "
                "High bar: only assert passages that are doing real epistemic work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "passage_id": {
                        "type": "string",
                        "description": "A short unique ID for this passage, e.g. 'fisher_capitalist_realism_p12'. Leave blank to auto-generate."
                    },
                    "text": {
                        "type": "string",
                        "description": "The exact passage text."
                    },
                    "author": {
                        "type": "string",
                        "description": "Author's full name, e.g. 'Mark Fisher'"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the source work, e.g. 'Capitalist Realism'"
                    },
                    "grounds": {
                        "type": "string",
                        "description": (
                            "Comma-separated list of existing node IDs this passage grounds "
                            "or supports, e.g. 'reasoning integrity, simulated knowledge'. "
                            "Must match node IDs that already exist -- search first if unsure."
                        )
                    },
                    "weight": {
                        "type": "number",
                        "description": (
                            "0.0-1.0: how strongly this passage SUPPORTS the claim. "
                            "Evidential strength, not your credence in the claim itself. "
                            "Default 0.8."
                        )
                    },
                    "source": {
                        "type": "string",
                        "description": "The `source` field from the search_corpus result. Joins this passage to the corpus."
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": "The `chunk_id` from the search_corpus result. Pass it -- it becomes the passage's id and the exact reference back to the evidence."
                    }
                },
                "required": ["text", "author", "title"]
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
                        "description": (
                            "The type of relationship -- must be exactly one of: EXPRESSES, "
                            "SUPPORTS, APPLIES_TO, TENSIONS_WITH, REQUIRES, OPPOSES, INFORMS, "
                            "EMBODIES, PRODUCES, ASSERTED_BY."
                        )
                    },
                    "weight": {"type": "number", "description": "Credence 0.0 to 1.0. How confident are you this relationship is real?"},
                    "grounding": {
                        "type": "string",
                        "description": (
                            "One sentence of reasoning: WHY does this relationship hold? "
                            "Prose for a human to read. For the machine-checkable "
                            "reference, use grounded_in."
                        )
                    },
                    "provenance": {
                        "type": "string",
                        "description": (
                            "Where this came from. Must be exactly one of: 'stated' "
                            "(Sean said it in this conversation or wrote it -- use this "
                            "only when you can point at his actual words), 'confirmed' "
                            "(you proposed it and he agreed), 'inferred' (you worked it "
                            "out -- THE DEFAULT, and the honest choice for most of what "
                            "you assert). Do not claim 'stated' for something you "
                            "concluded; that launders your inference into his voice. "
                            "'derived' is computed for you and should not be passed."
                        )
                    },
                    "grounded_in": {
                        "type": "string",
                        "description": (
                            "Comma-separated evidence for this assertion: corpus "
                            "chunk_id values returned by search_corpus, and/or ids of "
                            "graph nodes it rests on. This is a REFERENCE, not a "
                            "sentence -- it is what makes the claim traceable later. "
                            "If everything here is itself an inference, provenance is "
                            "automatically recorded as 'derived'."
                        )
                    }
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
            result = graph.search(args["query"], node_type=args.get("node_type"))

        elif tool_name == "find_neglected":
            result = graph.find_neglected(limit=args.get("limit", 10))

        elif tool_name == "what_changed":
            result = graph.what_changed(since=args.get("since", ""))

        elif tool_name == "find_orphans":
            result = graph.find_orphans(max_degree=args.get("max_degree", 0))

        elif tool_name == "get_full_graph":
            result = graph.get_full_graph()

        elif tool_name == "search_corpus":
            from corpus import format_for_tool
            where = {}
            if args.get("provenance"):
                where["provenance"] = args["provenance"]
            if args.get("author"):
                where["author"] = args["author"]
            return format_for_tool(
                args["query"],
                n_results=args.get("n_results", 5),
                where=where or None,
            )

        elif tool_name == "assert_node":
            result = graph.assert_node(
                args["node_id"],
                args["node_type"],
                description=args.get("description", ""),
                weight=args.get("weight", 0.7),
                provenance=args.get("provenance", "inferred"),
                grounded_in=args.get("grounded_in", ""),
            )

        elif tool_name == "assert_passage":
            result = graph.assert_passage(
                passage_id=args.get("passage_id", ""),
                text=args["text"],
                author=args["author"],
                title=args["title"],
                grounds=args.get("grounds", []),   # str or list; normalised in graph.py
                weight=args.get("weight", 0.8),
                source=args.get("source", ""),
                chunk_id=args.get("chunk_id", ""),
            )

        elif tool_name == "assert_edge":
            result = graph.assert_edge(
                args["source"],
                args["target"],
                args["edge_type"],
                weight=args.get("weight", 0.7),
                grounding=args.get("grounding", ""),
                provenance=args.get("provenance", "inferred"),
                grounded_in=args.get("grounded_in", ""),
            )

        else:
            result = {"error": f"Unknown tool: '{tool_name}'"}

    except KeyError as e:
        result = {"error": f"Missing required parameter: {e}"}
    except Exception as e:
        result = {"error": str(e)}

    return json.dumps(result, indent=2)

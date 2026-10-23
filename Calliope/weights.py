"""
Calliope Weight Editor
======================

Sets credences on graph nodes and edges without making you write 143 separate
one-liners.

    python3 weights.py --export     # write weights_todo.yaml for you to fill in
    python3 weights.py --apply      # read it back and write it into the graph
    python3 weights.py --status     # how much is still unweighted

WHY THIS EXISTS
---------------
Calliope's system prompt leans hard on weights:

    "A value with weight 0.9 is bedrock -- hold Sean to it hard.
     A value with weight 0.4 is tentative -- probe it, don't punish it."

    "Always set weight and grounding when asserting edges -- these are not
     optional. A claim without a grounding is not a claim, it's a guess."

In the seed graph, 47 of 48 nodes and 95 of 95 edges had no weight, and no
edge had any grounding. Every one of those instructions was describing
machinery that wasn't connected to anything. `find_values` sorts by
`weight or 0.5`, so with all weights null it returned values in arbitrary
order while claiming to put bedrock first.

None of this can be auto-filled. A weight is your judgment about your own
commitment. Inventing them would be precisely the confident fabrication the
whole design exists to prevent -- so this tool asks you, it does not guess.

WHAT THE NUMBERS MEAN
---------------------
    0.9 - 1.0   Bedrock. You'd rebuild your life around this. Hold me to it.
    0.7 - 0.8   Firm. A real commitment, but you can imagine being argued out.
    0.5 - 0.6   Working belief. You act on it without having settled it.
    0.3 - 0.4   Tentative. You suspect it. Probe, don't enforce.
    0.1 - 0.2   Barely held. In the graph because it came up, not because
                you're committed to it.

Be honest rather than flattering. A graph where everything is 0.9 carries
exactly as little information as one where everything is null.

START WITH NODES. Edge weights matter less and there are twice as many of
them; you can leave the edge section empty and come back to it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import CalliopeGraph

TODO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_todo.yaml")

HEADER = """\
# Calliope weights -- fill in the numbers, then run: python3 weights.py --apply
#
#   0.9-1.0  bedrock        0.7-0.8  firm         0.5-0.6  working belief
#   0.3-0.4  tentative      0.1-0.2  barely held
#
# Leave a line blank or delete it to skip that item. Nothing is overwritten
# unless you give it a number. Re-run --export any time; already-weighted
# items show their current value.
#
# Edges also take an optional grounding after a '|' -- why you believe the
# relationship holds. Example:
#
#   reasoning integrity -OPPOSES-> sycophancy: 0.9 | sycophancy is agreement
#     purchased at the cost of accuracy, which is the thing itself
#
"""


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def export(graph: CalliopeGraph) -> None:
    G = graph.G
    lines = [HEADER]

    # -- Nodes, grouped by type so like is judged against like ---------------
    lines.append("\nNODES:\n")
    by_type: dict[str, list[str]] = {}
    for n in G.nodes:
        by_type.setdefault(G.nodes[n].get("type", "Unknown"), []).append(n)

    # Values and Beliefs first: they carry the most weight in her reasoning.
    priority = ["Value", "Belief", "Anti", "Goal", "Tension",
                "Concept", "Role", "Project", "Person"]
    ordered = ([t for t in priority if t in by_type]
               + sorted(t for t in by_type if t not in priority))

    for t in ordered:
        lines.append(f"\n  # --- {t} ---")
        for n in sorted(by_type[t]):
            cur = G.nodes[n].get("weight")
            desc = (G.nodes[n].get("description") or "").strip()
            if desc:
                lines.append(f"  # {desc[:100]}")
            lines.append(f'  "{_esc(n)}": {cur if cur is not None else ""}')

    # -- Edges ---------------------------------------------------------------
    lines.append("\n\nEDGES:\n")
    lines.append("  # Optional. Format:  source -TYPE-> target: weight | grounding")
    lines.append("  # Safe to leave entirely blank and revisit later.\n")

    edges = sorted(G.edges, key=lambda uv: (G.edges[uv].get("type", ""), uv[0], uv[1]))
    last_type = None
    for u, v in edges:
        e = G.edges[u, v]
        et = e.get("type", "?")
        if et != last_type:
            lines.append(f"\n  # --- {et} ---")
            last_type = et
        w = e.get("weight")
        g = e.get("grounding") or ""
        suffix = f" | {g}" if g else ""
        lines.append(f'  "{_esc(u)}" -{et}-> "{_esc(v)}": '
                     f'{w if w is not None else ""}{suffix}')

    with open(TODO_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    unweighted_n = sum(1 for n in G.nodes if G.nodes[n].get("weight") is None)
    unweighted_e = sum(1 for uv in G.edges if G.edges[uv].get("weight") is None)
    print(f"Wrote {TODO_PATH}")
    print(f"  {unweighted_n} nodes and {unweighted_e} edges still need a weight.")
    print("\nOpen it, fill in numbers, then run: python3 weights.py --apply")


# -- Parsing ------------------------------------------------------------------
# Deliberately hand-rolled rather than using a YAML library: the format is
# trivial, and requiring pyyaml for this would be a dependency you don't
# otherwise need.

def _split_quoted(line: str) -> tuple[str, str] | None:
    """Parse a '"name": value' line. Returns (name, rest) or None."""
    if not line.startswith('"'):
        return None
    end = 1
    while end < len(line):
        if line[end] == "\\":
            end += 2
            continue
        if line[end] == '"':
            break
        end += 1
    else:
        return None
    name = line[1:end].replace('\\"', '"').replace("\\\\", "\\")
    rest = line[end + 1:].lstrip()
    return name, rest


def _parse_weight(text: str) -> tuple[float | None, str]:
    """Split 'value | grounding' into (weight or None, grounding)."""
    grounding = ""
    if "|" in text:
        text, grounding = text.split("|", 1)
        grounding = grounding.strip()
    text = text.strip()
    if not text:
        return None, grounding
    try:
        return float(text), grounding
    except ValueError:
        return None, grounding


def apply(graph: CalliopeGraph) -> None:
    if not os.path.exists(TODO_PATH):
        print(f"No {TODO_PATH}. Run: python3 weights.py --export")
        sys.exit(1)

    G = graph.G
    section = None
    set_nodes = set_edges = 0
    skipped: list[str] = []

    with open(TODO_PATH, encoding="utf-8") as f:
        raw_lines = f.readlines()

    for lineno, raw in enumerate(raw_lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("NODES:"):
            section = "nodes"
            continue
        if line.upper().startswith("EDGES:"):
            section = "edges"
            continue

        parsed = _split_quoted(line)
        if not parsed:
            continue
        name, rest = parsed

        if section == "nodes":
            if not rest.startswith(":"):
                continue
            w, _ = _parse_weight(rest[1:])
            if w is None:
                continue
            if name not in G:
                skipped.append(f"line {lineno}: no such node {name!r}")
                continue
            G.nodes[name]["weight"] = max(0.0, min(1.0, w))
            set_nodes += 1

        elif section == "edges":
            # rest looks like:  -TYPE-> "target": 0.8 | grounding
            if not rest.startswith("-") or "->" not in rest:
                continue
            etype, after = rest[1:].split("->", 1)
            etype = etype.rstrip("-").strip()
            after = after.lstrip()
            tgt = _split_quoted(after)
            if not tgt:
                continue
            target, tail = tgt
            if not tail.startswith(":"):
                continue
            w, grounding = _parse_weight(tail[1:])
            if w is None and not grounding:
                continue
            if not G.has_edge(name, target):
                skipped.append(f"line {lineno}: no such edge {name!r} -> {target!r}")
                continue
            if w is not None:
                G.edges[name, target]["weight"] = max(0.0, min(1.0, w))
            if grounding:
                G.edges[name, target]["grounding"] = grounding
            set_edges += 1

    graph.save()
    print(f"Applied {set_nodes} node weights and {set_edges} edge updates.")
    if skipped:
        print(f"\n{len(skipped)} line(s) skipped:")
        for s in skipped[:15]:
            print(f"  {s}")
        if len(skipped) > 15:
            print(f"  ... and {len(skipped) - 15} more")
    status(graph)


def status(graph: CalliopeGraph) -> None:
    G = graph.G
    n_tot = G.number_of_nodes()
    e_tot = G.number_of_edges()
    n_w = sum(1 for n in G.nodes if G.nodes[n].get("weight") is not None)
    e_w = sum(1 for uv in G.edges if G.edges[uv].get("weight") is not None)
    e_g = sum(1 for uv in G.edges if G.edges[uv].get("grounding"))

    def bar(done: int, total: int) -> str:
        if not total:
            return ""
        filled = int(20 * done / total)
        return "[" + "#" * filled + "." * (20 - filled) + f"] {done}/{total}"

    print(f"\n  nodes weighted    {bar(n_w, n_tot)}")
    print(f"  edges weighted    {bar(e_w, e_tot)}")
    print(f"  edges grounded    {bar(e_g, e_tot)}")

    unweighted_values = [n for n in G.nodes
                         if G.nodes[n].get("type") in ("Value", "Belief")
                         and G.nodes[n].get("weight") is None]
    if unweighted_values:
        print(f"\n  Highest-impact gaps -- Values and Beliefs with no weight "
              f"({len(unweighted_values)}):")
        for n in sorted(unweighted_values)[:12]:
            print(f"    - {n}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Set credences on Calliope's graph.")
    ap.add_argument("--export", action="store_true", help="Write weights_todo.yaml.")
    ap.add_argument("--apply", action="store_true", help="Read it back into the graph.")
    ap.add_argument("--status", action="store_true", help="Show completion.")
    args = ap.parse_args()

    g = CalliopeGraph()
    if args.export:
        export(g)
    elif args.apply:
        apply(g)
    elif args.status:
        status(g)
    else:
        ap.print_help()
        status(g)

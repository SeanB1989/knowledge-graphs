"""
Calliope -- graph-update reporting
==================================

Never trust the model's own narration of what she did.

THE BUG THIS EXISTS TO FIX
--------------------------
The system prompt (loop.py) shows Calliope an example "Graph updated:" block
and asks her to fill it in whenever she calls assert_node or assert_edge. A 12B
model sometimes just completes that pattern in prose -- writing a confident,
correctly-formatted report of nodes and edges it never actually created.

Nothing checked for this. The raw model text was printed and returned as-is, so
the terminal showed graph updates that had never happened. It was caught by
noticing that data/graph.json had not been modified in weeks, despite several
such reports.

THE FIX
-------
Every real assert_* call is already visible to the tool loop as a tool result,
so the model's own account of its actions is redundant -- and, being redundant,
is safe to throw away entirely. So:

  1. strip whatever "Graph updated:" block the model wrote, and
  2. rebuild the report from tool calls that actually ran and actually came
     back with {"status": "ok"}.

A report produced this way cannot be wrong: it is derived from the writes
themselves, not from the model's description of them. main.py also uses
FAKE_GRAPH_REPORT_RE to spot the mismatch case -- a claimed update with no real
writes behind it -- and give her one chance to either make it real or drop it.

These functions are pure: text in, text out, no I/O and no model calls. That
makes them the easiest part of the system to reason about and to test.
"""

from __future__ import annotations

import json
import re

# Matches a trailing "---\nGraph updated: ..." block at the very end of a reply.
FAKE_GRAPH_REPORT_RE = re.compile(r"\n-{3,}\s*\nGraph updated:.*\Z", re.DOTALL)

# Only these tools write to the graph. Anything else cannot produce a report line.
WRITING_TOOLS = ("assert_node", "assert_edge", "assert_passage")


def _strip_fake_graph_report(text: str) -> str:
    """Remove the model's self-authored update block, whatever it claimed."""
    return FAKE_GRAPH_REPORT_RE.sub("", text).rstrip()


def record_graph_change(name: str, args: dict, result_json: str, out: list) -> None:
    """
    Append a true report line -- but only for a call that really wrote.

    Called once per tool result. `out` is the running list of confirmed
    changes for this turn; it is appended to in place.
    """
    if name not in WRITING_TOOLS:
        return
    try:
        parsed = json.loads(result_json)
    except json.JSONDecodeError:
        return
    if parsed.get("status") != "ok":
        return  # failed assertion -- nothing was actually written, no line

    if name == "assert_node":
        node = parsed.get("node") or {}
        out.append(
            f"- Added/updated node: {args.get('node_id')} "
            f"({node.get('type', args.get('node_type'))}, "
            f"weight: {node.get('weight', args.get('weight', 0.7))}) -- "
            f"{node.get('description') or args.get('description', '')}"
        )
    elif name == "assert_edge":
        edge = parsed.get("edge") or {}
        out.append(
            f"- Added/updated edge: {edge.get('source')} "
            f"--[{edge.get('type')}]--> {edge.get('target')} "
            f"(weight: {edge.get('weight')}) -- "
            f"grounding: {edge.get('grounding') or '(none given)'}"
        )
    elif name == "assert_passage":
        out.append(
            f"- Added passage: {args.get('title')} ({args.get('author')}) -- "
            f"passage_id {args.get('passage_id') or '(auto)'}"
        )


def _format_graph_report(changes: list) -> str:
    """Render confirmed changes in the same format the prompt asks for."""
    if not changes:
        return ""
    return "\n\n---\nGraph updated:\n" + "\n".join(changes) + "\n---"


def finalize(text: str, changes: list) -> str:
    """
    Every returned answer passes through here: strip whatever graph-update
    claim the model wrote itself, replace it with one built only from tool
    calls that actually ran and actually succeeded.
    """
    return _strip_fake_graph_report(text) + _format_graph_report(changes)

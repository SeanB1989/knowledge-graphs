"""
Calliope -- local CLI
=====================
Run:  python3 main.py

Requires: pip3 install openai chromadb pdfplumber sentence-transformers
LM Studio must be running with its local server enabled.

HOW IT WORKS
------------
1. LM Studio runs a local server at localhost:1234 implementing the OpenAI API
   specification -- a standard format, not a product. The `openai` Python
   library speaks that format, so it works against any server implementing it.

2. We send a system prompt (who Calliope is), the conversation, and a list of
   tool schemas. The model decides when and which tools to call.

3. When it calls a tool, we execute it against graph.json / the corpus index
   and feed the result back as a `tool` message. The model continues.

4. Repeat until the model responds without calling tools. That is the answer.

This is a ReAct loop (Reason + Act) over native function calling.


WHERE EVERYTHING LIVES
----------------------
This file holds the loop and the terminal interface -- the part that decides
what happens next. The machinery it stands on lives elsewhere:

    config.py        every tunable: model, URL, token budgets, DEBUG
    model_io.py      talking to the model: calls, streaming, truncation repair
    graph_report.py  verifying what Calliope claims she wrote to the graph
    loop.py          the system prompt -- who Calliope is and how she works
    tools.py         the tool schemas and their dispatch
    graph.py         reading and writing data/graph.json
    corpus.py        hybrid search over the reading corpus
    embeddings.py    turning text into vectors
    ingest.py        building the corpus index from PDFs and text files

Read config.py first if you want to change behaviour; read loop.py if you want
to change her character.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LM_STUDIO_URL,
    MODEL,
    MAX_TOOL_ITERATIONS,
    CONTEXT_LIMIT,
    TOOL_RESULT_CHAR_CAP,
    DEBUG,
)
from model_io import call_model, generate_final, stitch, verify_server_context
from graph_report import FAKE_GRAPH_REPORT_RE, record_graph_change, finalize

from graph import CalliopeGraph
from tools import execute_tool, tool_definitions_for_api
from loop import build_system_prompt, build_continuation_prompt, REQUIRED_TOOLS


# -- ReAct loop ---------------------------------------------------------------

def run_tool_loop(graph: CalliopeGraph, system_prompt: str, user_message: str,
                  history: list) -> str:
    tools = tool_definitions_for_api()

    messages = ([{"role": "system", "content": system_prompt}]
                + history
                + [{"role": "user", "content": user_message}])

    tools_called: set[str] = set()
    graph_changes: list = []
    graph_correction_tried = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        message, finish_reason = call_model(messages, tools=tools)
        content = message.content or ""

        if not message.tool_calls:
            missing = REQUIRED_TOOLS - tools_called
            if missing and iteration == 0:
                missing_str = " and ".join(f"`{t}`" for t in sorted(missing))
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content":
                    f"[PROTOCOL ENFORCEMENT] You must call {missing_str} using your "
                    f"tools before responding. Do it now."})
                continue

            # Content that finished cleanly is the answer -- but only if it
            # actually finished. v1 returned it regardless, which is how a
            # length-truncated reply reached the terminal as if complete.
            if content.strip() and finish_reason == "stop":
                claimed_update = bool(FAKE_GRAPH_REPORT_RE.search(content))
                if claimed_update and not graph_changes and not graph_correction_tried:
                    # She wrote a "Graph updated:" narration but never actually
                    # called assert_node/assert_edge -- nothing was saved. Give
                    # her one chance to make it real (or drop the false claim)
                    # instead of just quietly erasing what she said.
                    graph_correction_tried = True
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content":
                        "[PROTOCOL ENFORCEMENT] You wrote a \"Graph updated\" "
                        "section, but you never actually called assert_node or "
                        "assert_edge -- so nothing was saved to the graph. If the "
                        "update is real, call the tool now. If it was not "
                        "genuinely new, drop that section and just answer."})
                    continue
                final_text = finalize(content, graph_changes)
                print(f"\nCalliope: {final_text}\n")
                return final_text

            if content.strip() and finish_reason == "length":
                convo = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content":
                        "You were cut off by the output limit. Continue from exactly "
                        "where you stopped. Do not repeat anything or start over."},
                ]
                # announce=False: this call is silent so the stitched result can be
                # checked for a fake "Graph updated" block before anything prints.
                rest = generate_final(convo, announce=False)
                final_text = finalize(stitch(content, rest), graph_changes)
                print(f"\nCalliope: {final_text}\n")
                return final_text

            # Empty final turn: the known stall. Force one clean, tool-free
            # generation from the results already gathered.
            messages.append({"role": "assistant", "content": ""})
            messages.append({"role": "user", "content":
                "Write your full response now, in prose, using the graph and corpus "
                "results above. Do not call any tools."})
            out = generate_final(messages, announce=False)
            final_text = finalize(out, graph_changes) if out else out
            final_text = final_text or "(Calliope returned no text -- try rephrasing.)"
            print(f"\nCalliope: {final_text}\n")
            return final_text

        for tc in message.tool_calls:
            tools_called.add(tc.function.name)

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name,
                             "arguments": tc.function.arguments},
            } for tc in message.tool_calls],
        })

        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            print(f"  [tool] {_trace(tc.function.name, args)}", flush=True)
            result = execute_tool(graph, tc.function.name, args)
            record_graph_change(tc.function.name, args, result, graph_changes)

            if len(result) > TOOL_RESULT_CHAR_CAP:
                result = result[:TOOL_RESULT_CHAR_CAP] + "\n... [truncated]"

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        messages.append({"role": "user", "content": build_continuation_prompt(tools_called)})

    # Ran out of iterations while still calling tools. Force an answer rather
    # than returning whatever half-formed content the last turn happened to have.
    print("\n  [tool budget exhausted -- forcing a response]")
    messages.append({"role": "user", "content":
        "Stop calling tools. Write your full response now from what you already have."})
    out = generate_final(messages, announce=False)
    final_text = finalize(out, graph_changes)
    print(f"\nCalliope: {final_text}\n")
    return final_text


# -- REPL ---------------------------------------------------------------------

BANNER = """
+======================================+
|           C A L L I O P E            |
|      personal ontology engine        |
+======================================+
"""


def _trace(name: str, args: dict) -> str:
    """
    One readable line per tool call.

    v1 printed the first 120 characters of raw JSON, which put whichever key the
    model happened to serialise first at the front. For assert_edge that was
    usually `grounding` -- a full sentence -- so the line ran out before source
    and target, and the trace showed that an edge had been asserted while hiding
    WHICH edge. The most important thing on screen was the thing being cut off.
    """
    try:
        if name == "assert_edge":
            return (f"assert_edge  {args.get('source')} --[{args.get('edge_type')}]--> "
                    f"{args.get('target')}  (w={args.get('weight', '?')})")
        if name == "assert_node":
            return (f"assert_node  {args.get('node_id')} "
                    f"[{args.get('node_type')}] (w={args.get('weight', '?')})")
        if name == "assert_passage":
            return (f"assert_passage  {args.get('author')}, '{args.get('title')}' "
                    f"-> grounds: {args.get('grounds') or '(nothing)'}")
        if name == "search_corpus":
            bits = [repr(args.get("query", ""))[:60]]
            if args.get("provenance"):
                bits.append(f"provenance={args['provenance']}")
            return f"search_corpus  {'  '.join(bits)}"
        if not args:
            return f"{name}()"
        return f"{name}  " + "  ".join(f"{k}={repr(v)[:40]}" for k, v in args.items())
    except Exception:
        return f"{name}(...)"


def keep_exchange(history: list, mine_only: bool = False) -> str:
    """
    Write the last exchange to the corpus so it stops evaporating.

    Sean's conversations with Calliope are his own writing -- reasoned, in his
    voice, generated continuously -- and until now every one of them was lost at
    `exit`. His writing is about 2.5% of the corpus by chunk count, and this is
    the one source of it that grows on its own.

    Two destinations, and the split matters:

      keep      -> corpus_dialogue/, provenance "dialogue". Both voices, labelled.
                   Retrievable as a record of what was already worked through.
      keep me   -> corpus_sean/, provenance "sean". Sean's turn ONLY.

    Calliope's output never lands in corpus_sean. Feeding a model's own words
    back in as its author's position is the exact failure his graph already
    names: "simulated knowledge -- the counterfeit of genuine expertise,
    indistinguishable in texture, dangerous at scale."
    """
    import re
    from datetime import datetime

    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    last_reply = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), None)
    if not last_user:
        return "Nothing to keep yet -- no exchange in this session."

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", last_user.lower())[:40].strip("_") or "exchange"
    pretty = datetime.now().strftime("%d %B %Y")

    if mine_only:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_sean")
        name = f"sean_note_{stamp}_{slug}.txt"
        body = (f"Sean Bruce -- note from conversation, {pretty}\n"
                f"{'=' * 60}\n\n{last_user}\n")
        where = "corpus_sean (provenance: sean)"
    else:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_dialogue")
        name = f"dialogue_{stamp}_{slug}.txt"
        body = (f"Conversation with Calliope, {pretty}\n"
                f"{'=' * 60}\n\nSEAN:\n{last_user}\n\n"
                f"CALLIOPE:\n{last_reply or '(no reply recorded)'}\n")
        where = "corpus_dialogue (provenance: dialogue)"

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write(body)

    words = len(body.split())
    return (f"Saved to {where}: {name} ({words} words)\n"
            f"  Run `python3 ingest.py` to make it searchable.")


def main():
    print(BANNER)

    graph = CalliopeGraph()
    summary = graph.get_full_graph()
    print(f"Graph loaded: {summary['node_count']} nodes, {summary['edge_count']} edges")
    print(f"Model: {MODEL} via {LM_STUDIO_URL}")
    print(f"Context budget: {CONTEXT_LIMIT} tokens")

    try:
        call_model([{"role": "user", "content": "ping"}], max_tokens=8)
        print("LM Studio: connected")
    except Exception as e:
        print(f"\nCould not connect to LM Studio at {LM_STUDIO_URL}")
        print("Make sure LM Studio is running with the local server enabled.")
        print(f"Error: {e}")
        sys.exit(1)

    # Outside the connection try-block on purpose: a diagnostic must never be
    # able to report itself as a connection failure.
    verify_server_context()

    try:
        import corpus
        corpus._load()
        print(f"Corpus: {len(corpus._docs)} chunks indexed\n")
    except Exception as e:
        print(f"Corpus: UNAVAILABLE -- {e}\n")

    system_prompt = build_system_prompt()
    history: list = []

    print("Type your message. Commands: 'graph', 'corpus <query>', 'keep', "
          "'keep me', 'changed', 'reset', 'exit'.")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        low = user_input.lower()

        if low == "exit":
            print("Goodbye.")
            break

        if low == "reset":
            history = []
            print("Conversation history cleared.")
            continue

        if low in ("keep", "keep me"):
            print(keep_exchange(history, mine_only=(low == "keep me")))
            continue

        if low == "changed" or low.startswith("changed since "):
            since = user_input[14:].strip() if low.startswith("changed since ") else ""
            report = graph.what_changed(since=since)
            print(f"\nChanges {report['since']}: {report['total_events']} event(s)")
            for e in report["events"]:
                print(f"  {e['date']}  {e['kind']:5} {e['event']:28} {e['what']}")
            if report.get("note"):
                print(f"\n  {report['note']}")
            print()
            continue

        if low == "graph":
            from collections import Counter
            summary = graph.get_full_graph()
            print(f"\nGraph: {summary['node_count']} nodes, {summary['edge_count']} edges")
            types = Counter(n.get("type") for n in summary["nodes"])
            for t, count in sorted(types.items(), key=lambda kv: str(kv[0])):
                nodes = [n["id"] for n in summary["nodes"] if n.get("type") == t]
                print(f"  {str(t):12} ({count}): {', '.join(nodes)}")
            continue

        if low.startswith("corpus "):
            try:
                import corpus
                out = corpus.search_corpus(user_input[7:], n_results=5)
                print(f"\n{out['diagnostics']['verdict']}  "
                      f"[{out['diagnostics']['scoring']}]")
                for i, r in enumerate(out["results"], 1):
                    print(f"\n[{i}] {r['citation']}  "
                          f"({r['confidence']}, {r['relevance']}, via {r['found_by']})")
                    print(f"    {r['text'][:280]}...")
                print()
            except Exception as e:
                print(f"Corpus error: {e}")
            continue

        print("\nCalliope is thinking...", flush=True)
        try:
            response = run_tool_loop(graph, system_prompt, user_input, history)

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            # Keep only the last few exchanges: conversational continuity
            # without unbounded context growth.
            history[:] = history[-6:]

            graph = CalliopeGraph()  # reload in case Calliope asserted new nodes
            print("-" * 50)
        except Exception as e:
            print(f"\n[Error]: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            print("Check that LM Studio is still running.\n")


if __name__ == "__main__":
    main()

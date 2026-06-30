"""
Calliope -- local CLI
Run this in Terminal: python3 main.py

Requires: pip install openai
LM Studio must be running with Gemma 4 loaded.

HOW IT WORKS:

1. LM Studio runs a local server at localhost:1234.
   It implements the OpenAI API specification -- a standard format, not
   a product. The openai Python library speaks this format, so it works
   with any server that implements it, including local ones.

2. We send the model a system prompt (who Calliope is) and a user message,
   plus a list of tool schemas (find_values, find_tensions, etc.).
   The model decides when and which tools to call natively.

3. When the model calls a tool, we execute it against graph.json and
   feed the result back as a tool message. The model continues.

4. We repeat until the model responds without calling any tools.
   That's the final answer.

This is a ReAct loop (Reason + Act) using the OpenAI native function
calling API -- no XML parsing, no regex, no leaked tags.
"""

import json
import os
import sys

# -- Dependency check ---------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency. Run: pip install openai")
    sys.exit(1)

# -- Import local modules -----------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import CalliopeGraph
from tools import execute_tool, tool_definitions_for_api
from loop import (
    build_system_prompt,
    build_continuation_prompt,
    REQUIRED_TOOLS,
)

# -- Config -------------------------------------------------------------------
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL = "google/gemma-4-12b-qat"
MAX_TOOL_ITERATIONS = 8

# -- Client -------------------------------------------------------------------
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")


def call_model(messages: list, tools: list = None):
    """
    Send messages to the model and return the full response message object.
    If tools are provided, the model can call them natively.
    """
    kwargs = dict(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=3500,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


def run_tool_loop(graph: CalliopeGraph, system_prompt: str, user_message: str) -> str:
    """
    The core ReAct loop using native function calling.

    - Model receives tools as structured schemas via the API
    - When it calls a tool, response.tool_calls is populated (not content)
    - We execute each tool, add results as role=tool messages
    - Loop until no tool calls -- that's the final response
    """
    tools = tool_definitions_for_api()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    tools_called: set = set()
    last_content = ""

    for iteration in range(MAX_TOOL_ITERATIONS):
        message = call_model(messages, tools=tools)
        content = message.content or ""
        last_content = content

        if not message.tool_calls:
            # No tool calls -- check required ones weren't skipped
            missing = REQUIRED_TOOLS - tools_called
            if missing and iteration == 0:
                missing_str = " and ".join(f"`{t}`" for t in sorted(missing))
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"[PROTOCOL ENFORCEMENT] You must call {missing_str} "
                        f"using your graph tools before responding. Do it now."
                    )
                })
                continue
            return content

        # Track which tools have been called
        for tc in message.tool_calls:
            tools_called.add(tc.function.name)

        # Add the assistant message (with tool calls) to history
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
        })

        # Execute each tool and add results
        for tc in message.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(graph, tool_name, args)
            print(f"  [graph] {tool_name}({tc.function.arguments})", flush=True)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # Continuation prompt
        continuation = build_continuation_prompt(tools_called)
        messages.append({"role": "user", "content": continuation})

    return last_content


# -- REPL ---------------------------------------------------------------------

def main():
    print("\n+======================================+")
    print("|           C A L L I O P E            |")
    print("|      personal ontology engine        |")
    print("+======================================+\n")

    graph = CalliopeGraph()
    summary = graph.get_full_graph()
    print(f"Graph loaded: {summary['node_count']} nodes, {summary['edge_count']} edges")
    print(f"Model: {MODEL} via {LM_STUDIO_URL}\n")

    # Test connection
    try:
        call_model([{"role": "user", "content": "ping"}])
        print("LM Studio: connected\n")
    except Exception as e:
        print(f"Could not connect to LM Studio at {LM_STUDIO_URL}")
        print(f"Make sure LM Studio is running with the server enabled.")
        print(f"Error: {e}")
        sys.exit(1)

    system_prompt = build_system_prompt()

    print("Type your message. Calliope will query the graph before responding.")
    print("Commands: 'graph' to inspect the graph, 'exit' to quit.\n")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        if user_input.lower() == "graph":
            summary = graph.get_full_graph()
            print(f"\nGraph: {summary['node_count']} nodes, {summary['edge_count']} edges")
            from collections import Counter
            types = Counter(n["type"] for n in summary["nodes"])
            for t, count in sorted(types.items()):
                nodes = [n["id"] for n in summary["nodes"] if n["type"] == t]
                print(f"  {t:12} ({count}): {', '.join(nodes)}")
            continue

        print("\nCalliope is thinking...", flush=True)
        try:
            response = run_tool_loop(graph, system_prompt, user_input)
            graph = CalliopeGraph()  # reload in case Calliope asserted new nodes
            print(f"\nCalliope: {response}\n")
            print("-" * 50)
        except Exception as e:
            print(f"\n[Error]: {e}")
            import traceback
            traceback.print_exc()
            print("Check that LM Studio is still running.\n")


if __name__ == "__main__":
    main()

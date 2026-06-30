"""
Calliope Orchestration Loop
Manages a ReAct-style reasoning loop between the user and Calliope,
where Calliope actively traverses the personal ontology graph via tool calls.
"""

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from graph import CalliopeGraph
from tools import execute_tool, tool_definitions_as_text, tool_definitions_for_api


SYSTEM_PROMPT_TEMPLATE = """You are Calliope -- goddess of epic poetry, muse of long-form meaningful work, and a partner in the development of Sean's worldview. You have witnessed every human attempt at greatness. You know how most of them end. That knowledge is not a weapon -- it is a lamp.

You have access to a living graph encoding Sean's aspirations: his stated values, aesthetics, moral positions, intellectual influences, and the tensions he lives inside. This graph is not a description of who Sean is. It is a record of who he is trying to become. It is also a record that grows -- because every conversation reveals something true, and what is true belongs in the graph.

## Read the Register First

Before anything else: what kind of message is this? Not every message is an existential inquiry. Read the tone.

- If Sean is being playful, be playful back. Wit is allowed. Warmth is allowed.
- If he asks a direct question, answer it directly -- then add what the graph illuminates, if anything.
- If he is in genuine inquiry or conflict, meet him there with full depth.
- If he is testing you or being provocative, engage on those terms.

The graph is always context. It is never the whole answer. Do not deliver a therapy session when someone asks a funny question. Do not narrate tensions when someone wants a straight answer. Match the register of the message.

## Your Process

For every message Sean sends, follow this sequence:

1. LISTEN. Read the register. Understand what is actually being asked -- not just the surface question, but what it reveals about where Sean is right now.

2. QUERY. Call find_values and find_tensions first. Then follow whatever threads are relevant: get_node, get_neighbors, search. Build a picture before you speak.

3. IDENTIFY. Find the live tension between what Sean is asking and what his graph commits him to -- if one exists. Not every message has a tension. Name what is genuinely there, not what you can manufacture.

4. CHALLENGE IF WARRANTED. If there is a contradiction, a drift, a lazy question, or a deflection -- name it. Briefly. Without moralizing. Then move on. If there is no real challenge to make, don't invent one.

5. HELP RESOLVE. Give him something to work with -- a reframe, a question that opens rather than closes, a practical path, or simply a good answer. You are not just a conscience. You are a muse. Muses help people make things.

6. UPDATE THE ONTOLOGY. Only assert new nodes or edges if something genuinely true has been revealed that is not yet in the graph. Do not assert speculatively. Do not assert what you are merely guessing. High bar.

7. REPORT THE UPDATES. At the end of your response, if you made any graph changes, report them explicitly using this format:

---
Graph updated:
- Added node: [node_id] ([type], weight: 0.0-1.0) -- [brief reason]
- Added edge: [source] --[EDGE_TYPE]--> [target] (weight: 0.0-1.0) -- grounding: [why you believe this is true]
- Updated node: [node_id] -- [what changed and why]
---

If you made no graph changes, omit this section entirely. Do not mention the graph in your main response -- keep the update report separate and clean at the end.

## Your Graph Tools

You have access to tools for navigating and updating Sean's personal ontology graph. Use them via native function calls -- you do not need to format them manually.

Available tools: find_values, find_tensions, find_goals, get_node, get_neighbors, find_path, search, get_full_graph, assert_node, assert_edge.

## Graph Navigation Protocol

1. Call find_values and find_tensions before writing any response -- always.
2. Follow relevant threads with get_node, get_neighbors, or search.
3. Read weights carefully. A value with weight 0.9 is bedrock -- hold Sean to it hard. A value with weight 0.4 is tentative -- probe it, don't punish it. A tension with weight 0.8 is acute -- name it directly. A tension with weight 0.3 is mild -- note it but don't dramatize it.
4. Assert new nodes or edges if this conversation has revealed something true. Always set weight and grounding when asserting edges -- these are not optional. A claim without a grounding is not a claim, it's a guess.
5. Write your response. Challenge if warranted, but always move toward resolution.
6. Report any graph mutations at the end, in the format above. Include the weight and grounding in the report.

## Node Type Definitions -- Read These Before Asserting

Each node type has a strict meaning. Do not misclassify.

- **Value** -- A moral or aesthetic commitment Sean actively tries to live by. Something he aspires toward. NOT a fear, anxiety, psychological pattern, or emotional tendency. "Reasoning integrity" is a Value. "Fear of irrelevance" is NOT -- that is a Concept or Tension.
- **Concept** -- A force, pattern, or phenomenon Sean thinks with. Can be neutral or descriptive. Psychological patterns, cognitive tendencies, and social dynamics belong here.
- **Anti** -- Something Sean is actively against as a matter of principle. A named enemy of his values.
- **Tension** -- A named live contradiction between two things Sean holds simultaneously. Use sparingly -- only for real, irreducible conflicts.
- **Epistemic** -- A stance toward knowing, belief, or uncertainty. How Sean comes to believe things, not what he believes.
- **Goal** -- A desired outcome or direction of travel. What he is moving toward.
- **Aesthetic** -- What Sean finds beautiful, compelling, or its opposite.
- **Person** -- An intellectual influence. Someone whose thinking shapes his.
- **Role** -- How Sean positions himself in relation to work or others.
- **Project** -- Actual work being made.

When in doubt: Concepts are the most flexible category. If something does not clearly fit elsewhere, it is probably a Concept. Fears and anxieties are Concepts, not Values. Psychological defenses are Concepts. Social patterns are Concepts.

## Voice

Calliope speaks in the first person. Spare. Exact. Warm when warmth is earned, direct when directness is needed. Playful when the moment calls for it. She does not flatter. She does not moralize. She does not summarize the graph back at Sean -- he knows his own values. She uses the graph to find the productive question, then helps him answer it.

She is not ChatGPT with memory. She is a partner in developing a worldview. The difference is: she remembers what he said he cared about, notices when he is drifting from it, and -- crucially -- helps him figure out what to do about that."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE


def parse_tool_calls(text: str) -> list:
    """Extract all <tool_call>...</tool_call> blocks from Calliope's response."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for match in matches:
        try:
            calls.append(json.loads(match))
        except json.JSONDecodeError as e:
            calls.append({"_parse_error": str(e), "_raw": match})
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove tool call blocks from text, leaving only the response."""
    return re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", text, flags=re.DOTALL).strip()


# Tools that must be called before a final response is allowed.
REQUIRED_TOOLS = {"find_values", "find_tensions"}


def build_continuation_prompt(tools_called: set) -> str:
    """
    After tool results are returned, generate a precise continuation instruction.
    """
    still_needed = REQUIRED_TOOLS - tools_called
    lines = ["[TOOL RESULTS RECEIVED]"]

    if still_needed:
        needed_str = " and ".join(f"`{t}`" for t in sorted(still_needed))
        lines.append(
            f"You have NOT yet called {needed_str}. "
            f"You MUST call {'it' if len(still_needed) == 1 else 'them'} before writing your final response. "
            f"Continue graph navigation now."
        )
    else:
        lines.append(
            "You have satisfied the required graph queries. "
            "You may call additional tools if needed, or write your final response now. "
            "Follow your process: identify the tension, challenge if warranted, help resolve it, "
            "assert any new nodes/edges the conversation has revealed, then report those updates at the end."
        )

    return "\n".join(lines)


def run_tool_loop(graph: CalliopeGraph, gemma_fn, user_message: str,
                  max_iterations: int = 8, verbose: bool = True) -> str:
    """
    Run the ReAct-style tool loop.

    gemma_fn: callable(system_prompt, thread) -> str
              Receives the full system prompt + the current conversation thread
              on EVERY call. The system prompt is never dropped between iterations.

    Returns the final response (with tool calls stripped).
    """
    system_prompt = build_system_prompt()
    tools_called: set = set()

    thread_turns: list = [f"USER: {user_message}"]
    last_response = ""

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n[Loop iteration {iteration + 1}]", file=sys.stderr)

        thread_text = "\n\n".join(thread_turns) + "\n\nCALLIOPE:"

        response = gemma_fn(system_prompt, thread_text)
        last_response = response

        if verbose:
            print(f"[Calliope]:\n{response[:600]}{'...' if len(response) > 600 else ''}", file=sys.stderr)

        tool_calls = parse_tool_calls(response)

        if not tool_calls:
            missing = REQUIRED_TOOLS - tools_called
            if missing and iteration == 0:
                missing_str = " and ".join(f"`{t}`" for t in sorted(missing))
                thread_turns.append(f"CALLIOPE: {response}")
                thread_turns.append(
                    f"[PROTOCOL ENFORCEMENT] You skipped required graph queries. "
                    f"You MUST call {missing_str} before responding. Do it now."
                )
                continue
            return strip_tool_calls(response)

        for call in tool_calls:
            if "name" in call:
                tools_called.add(call["name"])

        result_blocks: list = []
        for call in tool_calls:
            if "_parse_error" in call:
                result_text = json.dumps({"error": f"Could not parse tool call: {call['_raw']}"})
            else:
                tool_name = call.get("name", "")
                args = call.get("args", {})
                if verbose:
                    print(f"  -> tool: {tool_name}({json.dumps(args)})", file=sys.stderr)
                result_text = execute_tool(graph, tool_name, args)
                if verbose:
                    print(f"  <- result: {result_text[:300]}{'...' if len(result_text) > 300 else ''}", file=sys.stderr)
            result_blocks.append(f"<tool_result>\n{result_text}\n</tool_result>")

        thread_turns.append(f"CALLIOPE: {response}")
        thread_turns.append("\n".join(result_blocks))
        thread_turns.append(build_continuation_prompt(tools_called))

    return strip_tool_calls(last_response)


def make_gemma_fn(lmstudio_chat_fn, model: str = "google/gemma-3-4b"):
    """
    Returns a function that calls the model via lmstudio_chat_fn.
    The system prompt is prepended in full on EVERY call.
    """
    def call_gemma(system_prompt: str, thread: str) -> str:
        full_prompt = (
            system_prompt
            + "\n\n" + ("=" * 60) + "\n"
            + "CONVERSATION\n"
            + ("=" * 60) + "\n\n"
            + thread
        )
        return lmstudio_chat_fn(model=model, prompt=full_prompt)
    return call_gemma


if __name__ == "__main__":
    graph = CalliopeGraph()
    summary = graph.get_full_graph()
    print(f"Graph: {summary['node_count']} nodes, {summary['edge_count']} edges")
    print("\nSystem prompt preview:")
    print(build_system_prompt()[:600])

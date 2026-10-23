"""
Calliope -- talking to the model
================================

Everything between "we have a list of messages" and "we have text back".
Token accounting, context pruning, the actual API calls, streaming, and the
repair logic for replies that get cut off mid-sentence.

Nothing in here knows anything about the graph, the corpus, or Calliope's
personality. It moves messages in and text out.


================================================================================
THE TRUNCATION BUG, AND WHAT WAS DONE ABOUT IT
================================================================================

"She cuts off mid-response" has three possible causes, and v1 could not tell
them apart because it never looked at the one field that says which:

  finish_reason == "stop"    the model chose to end. Normal.
  finish_reason == "length"  it hit the output cap mid-sentence. TRUNCATION.
  finish_reason == None/other the server gave up (usually context overflow).

v1 read `response.choices[0].message` and threw the rest away, including
`finish_reason` and `usage`. So a truncated answer and a finished answer were
literally indistinguishable to the program. Everything below follows from
fixing that.

  1. FINISH REASON IS NOW CHECKED. On "length", we don't just give up: we ask
     the model to continue from where it stopped and stitch the pieces
     together, so long answers complete instead of dangling.

  2. TOKEN USAGE IS LOGGED. Every call prints prompt/completion tokens. This
     matters more than it sounds, because of (3).

  3. CONTEXT OVERFLOW IS THE LIKELIER CULPRIT. LM Studio has a per-model
     context length slider that commonly defaults to 4096 tokens. Calliope's
     system prompt alone is ~2,400. Add tool results -- a corpus search used to
     return five 400-word passages, ~2,000 tokens by itself -- and a single
     turn blows the window. When that happens the server silently drops the
     oldest messages (which is the SYSTEM PROMPT, from the front) and generation
     degrades or stops early. It looks exactly like the model "cutting off."

     So: set LM Studio's context length to at least 16384 for this model. The
     program now warns you when a prompt is approaching CONTEXT_LIMIT, and
     prunes its own history to stay under it.

  4. TOOL RESULTS ARE NOW BUDGETED. Old tool results get collapsed to stubs
     once they're several turns back, and corpus passages are trimmed before
     they enter context. The model does not need the full text of a passage it
     already read three iterations ago.

  5. max_tokens RAISED to 4096, and the final answer is STREAMED so you can
     watch it arrive rather than staring at a blank terminal wondering if it
     has hung.
"""

from __future__ import annotations

import json

from config import (
    client,
    LM_STUDIO_URL,
    MODEL,
    MAX_TOKENS,
    MAX_CONTINUATIONS,
    CONTEXT_LIMIT,
    CONTEXT_WARN_AT,
    KEEP_FULL_TOOL_RESULTS,
    DEBUG,
)

_warned_about_context = False


# -- Token accounting ---------------------------------------------------------

def verify_server_context(verbose: bool = True) -> dict:
    """
    Ask LM Studio what context it ACTUALLY allocated, and compare.

    WHY THIS EXISTS
    ---------------
    config.py has always carried the comment "Must match the context length
    configured for this model in LM Studio" -- an invariant that nothing
    enforced. On 2026-08-13 it turned out CONTEXT_LIMIT had been 16384 while
    LM Studio had the model loaded at 8192, for an unknown length of time.
    Every conversation ran at half its assumed budget: prune_history kept
    history it had no room for, and the server truncated the rest silently.
    There was no error, no warning, and no way to notice from the outside.

    This uses LM Studio's NATIVE endpoint (/api/v1/models), not the
    OpenAI-compatible one, because only the native one reports the context
    length each loaded instance was actually given.

    Advisory, never fatal: an older LM Studio without this endpoint, or a
    server that is simply down, must not stop Calliope from starting. The
    point is to make a silent mismatch loud, not to add a new way to fail.
    """
    import json as _json
    import urllib.request

    try:
        return _verify_server_context(verbose)
    except Exception as e:
        # This check is a diagnostic. If it ever raises -- a missing import, a
        # response shape LM Studio changed, anything -- it must not stop
        # Calliope from running. It shipped once with exactly that fault and
        # a NameError in here presented to the user as "could not connect to
        # LM Studio", which is the most misleading possible failure.
        if DEBUG:
            print(f"  [context check failed: {type(e).__name__}: {e}]")
        return {"checked": False, "reason": f"{type(e).__name__}: {e}"}


def _verify_server_context(verbose: bool = True) -> dict:
    import json as _json
    import urllib.request

    base = LM_STUDIO_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = base.rstrip("/") + "/api/v1/models"

    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            data = _json.loads(r.read().decode())
    except Exception as e:
        if verbose and DEBUG:
            print(f"  [context check skipped: {e}]")
        return {"checked": False, "reason": str(e)}

    entries = data.get("models", data if isinstance(data, list) else [])
    instances = []
    for m in entries:
        for inst in (m.get("loaded_instances") or []):
            instances.append({
                "id": inst.get("id") or m.get("key"),
                "key": m.get("key"),
                "context": (inst.get("config") or {}).get("context_length"),
                "type": m.get("type"),
            })

    llm = [i for i in instances if i.get("type") != "embedding"]
    mine = [i for i in llm if i["id"] == MODEL or i["key"] == MODEL]

    out = {"checked": True, "loaded": llm, "matched": mine}

    if not mine:
        ids = ", ".join(sorted({i["id"] for i in llm})) or "none"
        print(f"\n  !! CALLIOPE_MODEL is '{MODEL}' but no loaded instance has that id.")
        print(f"     Loaded right now: {ids}")
        print(f"     Set CALLIOPE_MODEL to one of those, or load the right model.")
        return out

    # More than one instance answering to this model is its own hazard: you
    # cannot tell which one a request lands on, so they must not differ.
    same_key = [i for i in llm if i["key"] == mine[0]["key"]]
    if len(same_key) > 1:
        ctxs = {i["context"] for i in same_key}
        print(f"\n  !! {len(same_key)} instances of '{mine[0]['key']}' are loaded "
              f"(contexts: {sorted(c for c in ctxs if c)}).")
        print(f"     Requests may land on either. Eject the duplicates in LM Studio.")

    actual = mine[0]["context"]
    # The two directions are not equally bad, so they do not get the same alarm.
    # Budgeting for MORE than the server has means silent truncation at the far
    # end of a long conversation -- a real fault. Budgeting for LESS just wastes
    # room, and LM Studio's allocation rarely lands on the round number you set
    # (a 32768 slider reports 32651), so treating any difference as an error
    # would cry wolf on every startup.
    if actual and actual < CONTEXT_LIMIT:
        print(f"\n  !! CONTEXT OVER-BUDGET")
        print(f"     Calliope is budgeting for : {CONTEXT_LIMIT:,} tokens")
        print(f"     LM Studio actually loaded : {actual:,} tokens")
        print(f"     The server will truncate silently. Either raise the context in")
        print(f"     LM Studio and reload, or run: CALLIOPE_CONTEXT={actual} python3 main.py")
    elif actual and actual - CONTEXT_LIMIT > 1024:
        print(f"  Note: LM Studio has {actual:,} tokens loaded, Calliope is using "
              f"{CONTEXT_LIMIT:,}.")
        print(f"        {actual - CONTEXT_LIMIT:,} tokens spare -- "
              f"CALLIOPE_CONTEXT={actual} to claim them.")
    elif actual and verbose:
        print(f"  Context verified: {actual:,} tokens allocated, budgeting {CONTEXT_LIMIT:,}.")

    out["actual"] = actual
    return out


def approx_tokens(messages: list) -> int:
    """
    Rough token estimate: ~4 characters per token for English. Used only to
    decide when to prune, before we have real numbers back from the server.
    """
    chars = 0
    for m in messages:
        chars += len(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            chars += len(json.dumps(tc))
    return chars // 4


def prune_history(messages: list) -> list:
    """
    Keep context under control without losing the thread.

    Invariants:
      * The system prompt is never dropped. If the server has to truncate, it
        truncates from the front -- which is precisely the worst thing to lose.
      * Every assistant message with tool_calls keeps its matching tool
        results, because the OpenAI message format requires that pairing and
        the API errors out if it's broken.

    Strategy: collapse the CONTENT of older tool results to a stub. The model
    keeps the structural record that it called the tool, without paying full
    context price for output it has already reasoned over.
    """
    tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_positions) <= KEEP_FULL_TOOL_RESULTS:
        return messages

    stale = set(tool_positions[:-KEEP_FULL_TOOL_RESULTS])
    out = []
    for i, m in enumerate(messages):
        if i in stale:
            body = str(m.get("content") or "")
            m = dict(m)
            m["content"] = (
                f"[earlier result, {len(body)} chars, collapsed to save context. "
                f"Preview: {body[:180]}...]"
            )
        out.append(m)
    return out


def _check_context(prompt_tokens: int) -> None:
    global _warned_about_context
    if prompt_tokens > CONTEXT_LIMIT * CONTEXT_WARN_AT and not _warned_about_context:
        _warned_about_context = True
        print(
            f"\n  [!] Prompt is {prompt_tokens} tokens, against a configured limit of "
            f"{CONTEXT_LIMIT}.\n"
            f"      If Calliope truncates or returns nothing, the context window in "
            f"LM Studio\n"
            f"      is probably too small. Open the model's settings there, raise "
            f"'Context Length'\n"
            f"      to 16384 or more, reload the model, and set CALLIOPE_CONTEXT to match.\n",
            flush=True,
        )


# -- Model calls --------------------------------------------------------------

def call_model(messages: list, tools: list | None = None, max_tokens: int = MAX_TOKENS):
    """
    One non-streaming call. Returns (message, finish_reason).

    Unlike v1 this returns finish_reason, which is the whole point -- without
    it there is no way to know a response was cut off rather than completed.
    """
    kwargs = dict(model=MODEL, messages=prune_history(messages),
                  temperature=0.7, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]

    usage = getattr(response, "usage", None)
    if usage:
        _check_context(usage.prompt_tokens)
        if DEBUG:
            print(f"  [tokens] prompt={usage.prompt_tokens} "
                  f"completion={usage.completion_tokens} "
                  f"finish={choice.finish_reason}", flush=True)

    return choice.message, choice.finish_reason


MIN_OVERLAP = 10  # shorter than this and we risk deleting legitimate text


def stitch(a: str, b: str) -> str:
    """
    Join a truncated fragment to its continuation.

    Two failure modes to avoid, both of which the first version of this
    function actually hit:

    1. REPEATED TEXT. A model asked to continue usually restates the last
       clause first ("...It is between" / "It is between the part of you...").
       So we look for the longest suffix of `a` that is also a prefix of `b`
       and splice there.

       The overlap must be at least MIN_OVERLAP characters AND span a word
       boundary. Matching on something short like "the " would happily delete
       real text, which is worse than a small duplication.

    2. INSERTED SPACES MID-WORD. The output limit cuts at a token boundary,
       which lands mid-word more often than not ("...incompl" + "ete"). Naively
       joining with a space yields "incompl ete".

       The fix is to trust whatever leading whitespace the model emitted rather
       than guessing: a genuine new word arrives as " ete"-style with its own
       space. We only add one ourselves after sentence-ending punctuation,
       where a new word is certain and a mid-word split is impossible.
    """
    if not a:
        return b
    if not b:
        return a

    for candidate in (b, b.lstrip()):
        upper = min(300, len(a), len(candidate))
        for n in range(upper, MIN_OVERLAP - 1, -1):
            head = candidate[:n]
            if " " in head and a.endswith(head):
                return a + candidate[n:]

    if a[-1] in ".!?\"')]" and not b[:1].isspace():
        return a + " " + b
    return a + b


def generate_final(messages: list, stream: bool = True, announce: bool = True) -> str:
    """
    Produce the final prose answer, with tools DISABLED.

    Tools are off deliberately. Small local models sometimes get stuck in
    "function-calling mode" and emit empty turns. Removing the tools removes
    the option, so the only thing left to do is write the answer.

    If the answer hits the output cap, we resume it rather than returning a
    sentence that stops mid-clause.
    """
    full = ""
    convo = list(messages)

    for attempt in range(MAX_CONTINUATIONS + 1):
        if stream:
            piece, reason = _stream_once(convo, announce=announce)
        else:
            msg, reason = call_model(convo, tools=None)
            piece = msg.content or ""
            if announce:
                print(piece, end="", flush=True)

        full = stitch(full, piece) if full else piece

        if reason != "length":
            break

        if attempt == MAX_CONTINUATIONS:
            full += "\n\n[...response hit the length limit. Ask her to continue.]"
            break

        if announce:
            print("\n  [continuing -- hit output limit]\n", flush=True)
        convo = messages + [
            {"role": "assistant", "content": full},
            {"role": "user", "content":
                "You were cut off by the output limit. Continue from exactly where "
                "you stopped, mid-sentence if that is where it ended. Do not repeat "
                "anything you have already written and do not start over."},
        ]

    return full.strip()


def _stream_once(messages: list, announce: bool = True) -> tuple[str, str]:
    """Stream one completion, printing as it arrives. Returns (text, finish_reason)."""
    parts: list[str] = []
    reason = "stop"
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=prune_history(messages),
            temperature=0.7,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            choice = event.choices[0]
            delta = getattr(choice.delta, "content", None)
            if delta:
                parts.append(delta)
                if announce:
                    print(delta, end="", flush=True)
            if choice.finish_reason:
                reason = choice.finish_reason
    except Exception as e:
        # Streaming is a nicety, not a requirement. Fall back rather than die.
        if not parts:
            msg, reason = call_model(messages, tools=None)
            text = msg.content or ""
            if announce:
                print(text, end="", flush=True)
            return text, reason
        if announce:
            print(f"\n  [stream interrupted: {e}]", flush=True)

    return "".join(parts), reason

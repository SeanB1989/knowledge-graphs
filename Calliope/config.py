"""
Calliope -- configuration and the LM Studio client
==================================================

Single source of truth for every tunable in the system. Nothing in this file
does any work; it only decides how the rest of the program behaves. If you want
to point Calliope at a different model, change the context budget, or turn on
debug output, this is the only file you need to open.

Every value can be overridden with an environment variable, so you can try a
change without editing code at all:

    CALLIOPE_MODEL="google/gemma-4-27b" python3 main.py
    CALLIOPE_CONTEXT=32768 python3 main.py
    CALLIOPE_DEBUG=1 python3 main.py
"""

from __future__ import annotations

import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency. Run: pip3 install openai")
    sys.exit(1)


# -- Where the model lives ----------------------------------------------------

LM_STUDIO_URL = os.environ.get("CALLIOPE_LM_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("CALLIOPE_MODEL", "google/gemma-4-12b-qat")


# -- Budgets ------------------------------------------------------------------

# Raised 8 -> 6 when context was scarce, then 6 -> 14 on 2026-08-13 once the
# window went from 8,192 to 32,651. At 6 this cut Calliope off mid-flow while
# she was asserting edges: find_orphans, find_values, find_tensions,
# search_corpus and four assert_edge calls exhausted the budget before she
# could finish or report. Multi-step graph work is the point of the loop, and
# 6 iterations is barely enough to look before writing.
MAX_TOOL_ITERATIONS = 14
MAX_TOKENS = 4096            # output cap per call
MAX_CONTINUATIONS = 3        # how many times we'll resume a length-truncated reply

# Must match the context length configured for this model in LM Studio.
# If you raise the slider there, raise this too (or set CALLIOPE_CONTEXT).
#
# This is no longer taken on trust. main.py calls model_io.verify_server_context()
# at startup, which asks LM Studio's native /api/v1/models what it actually
# allocated and complains loudly if the two disagree.
#
# Why that check exists: on 2026-08-13 this value was 16384 while LM Studio had
# the model loaded at 8192. Against a ~5,500-token fixed cost for the system
# prompt and tool schemas, that left roughly 2,600 tokens of working room --
# less than three corpus results -- and nothing anywhere reported it.
# 32651, not 32768: LM Studio reports what it actually allocated, which is a
# little under the round number on the slider. Budgeting for tokens that do not
# exist is the dangerous direction -- the server truncates and says nothing.
CONTEXT_LIMIT = int(os.environ.get("CALLIOPE_CONTEXT", "32651"))
CONTEXT_WARN_AT = 0.75       # warn once a prompt passes this fraction of the limit

# Keep full text for this many of the most recent tool results; older ones are
# collapsed to a one-line stub.
KEEP_FULL_TOOL_RESULTS = 4
TOOL_RESULT_CHAR_CAP = 4000  # hard cap on any single tool result entering context

DEBUG = os.environ.get("CALLIOPE_DEBUG", "").lower() in ("1", "true", "yes")


# -- The client ---------------------------------------------------------------
# LM Studio implements the OpenAI API specification -- a standard format, not a
# product -- so the official `openai` library talks to it unchanged. The api_key
# is required by the library and ignored by LM Studio.

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

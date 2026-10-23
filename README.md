# Calliope

**A personal ontology engine. A different kind of AI companion.**

Calliope is a local AI that knows what you care about — your values, your goals, the tensions you live inside, the thinkers who shaped you. Before it says anything, it reads the graph. It notices when you're drifting. It helps you figure out what to do about that.

It runs entirely on your machine, on an open-weights model via [LM Studio](https://lmstudio.ai/). Inference, graph storage, and retrieval run locally. Models and Python packages must be downloaded during setup; the optional cross-encoder reranker may download its weights on first use.

---

## What it is

Most AI assistants have no memory of who you are. They are, by design, general-purpose — optimised for the average user across the average query. Calliope is the opposite. It is a single-person instrument, built around a living knowledge graph that encodes your particular worldview: your stated values, your intellectual debts, your goals, the contradictions you haven't resolved.

Every response begins with a query against that graph. Calliope cannot respond without first checking what you care about and what tensions are currently live. This is enforced at the protocol level — it's not a suggestion in the system prompt, it's a required step in the reasoning loop.

The result is an AI that can hold you to account in a way a general assistant never can: because it knows what you said you were trying to become.

---

## How it works

Calliope uses a **ReAct loop** (Reason + Act) with native OpenAI-format function calling. The model receives a structured list of tools it can call against your graph, and is required to call `find_values` and `find_tensions` before writing any response.

```
User message
    ↓
find_values() + find_tensions()   ← required
    ↓
get_node(), search(), find_path() ← as needed
    ↓
[optional] assert_node() / assert_edge()  ← if something true was revealed
    ↓
Final response + graph update report
```

The graph grows. When a conversation reveals something genuinely true about you — a new tension, a clarified value, an intellectual influence you hadn't encoded — Calliope asserts it with a weight and a grounding. The ontology is a living record, not a static document.

Calliope runs locally against [LM Studio](https://lmstudio.ai/) via the OpenAI-compatible API. The default model is `google/gemma-4-12b-qat`, but any model LM Studio can serve will work.

---

## Design philosophy: fewer node types, richer edges

### The node taxonomy

Calliope uses a small core taxonomy, with **11 accepted node types** in the current implementation:

| Type | What it encodes |
|---|---|
| `Value` | Moral, practical, and aesthetic commitments you actively try to live by. Includes aesthetic sensibilities — the preferences that shape how you make and judge work. |
| `Belief` | Claims about how the world is or works. Includes both first-order worldview claims and second-order epistemic stances — how you come to know things, what you trust, how you revise. |
| `Concept` | The intellectual vocabulary you think *with*. Forces, patterns, frameworks, self-models. Tools for reasoning, not conclusions from it. |
| `Goal` | Desired outcomes and directions of travel. |
| `Tension` | Named live contradictions you hold simultaneously. Used sparingly — only for real, irreducible conflicts. |
| `Anti` | What you are against as a matter of principle. Named antagonists to your values — the things defined as wrong that are constitutive of what you stand for. |
| `Person` | Any individual with a significant constitutive relationship to your worldview: intellectual influences (thinkers whose ideas shape yours), formative relationships (parents, teachers, partners, mentors), and created characters (fictional persons you've made who now think alongside you). |
| `Project` | Actual work being made. |
| `Role` | A role or position within the personal ontology. |
| `Passage` | A cited corpus passage, linked to its source and the claims it grounds. |
| `Text` | A source work, linked to its author. |

This is a deliberately **small taxonomy**. Almost any object — a fear, a philosophy, a habit, a relationship — can be encoded within it, but it must be classified into one of these categories. That constraint is doing work.

`Belief` absorbs what other systems call *epistemic* nodes — stances toward knowing are beliefs, second-order ones. `Value` absorbs aesthetic commitments — aesthetic preferences function as values in the graph: they carry weight, guide decisions, and can tension against other things. `Concept` absorbs self-positioning models — how you understand your own role in your work and world is part of the conceptual vocabulary you think with. The consolidation isn't loss; it's clarity about what these things actually are.

### The edge vocabulary

The graph has **13 accepted edge types**:

| Edge | What it expresses |
|---|---|
| `SUPPORTS` | One node makes another more plausible or robust |
| `OPPOSES` | Direct antagonism between nodes |
| `TENSIONS_WITH` | Live contradiction — both are held, neither resolves the other |
| `REQUIRES` | One node depends on another to be coherent |
| `INFORMS` | One node shapes or contextualises another without determining it |
| `EMBODIES` | One node is a concrete expression or instantiation of another |
| `EXPRESSES` | One node gives form to what another contains abstractly |
| `PRODUCES` | One node generates another as output |
| `APPLIES_TO` | One node is relevant to or operates within the domain of another |
| `ASSERTED_BY` | A claim traced to a source — person, text, or experience |
| `GROUNDS` | A passage supports a graph claim |
| `SOURCED_FROM` | A passage comes from a source text |
| `AUTHORED_BY` | A text was authored by a person |

### Why this design?

The choice to keep node types *few* and edge types *rich* is not arbitrary. It reflects a theory about what makes a personal ontology useful for this kind of project.

**A small node taxonomy creates pressure toward generality.** When you cannot create a new node type for every distinction, you are forced to find where a new idea fits within existing categories. That friction produces clarity: is this really a *Value* I'm committing to, or is it a *Concept* I'm merely interested in? The discipline of classification is part of the epistemological work.

**Rich edge semantics is where the real meaning lives.** Two nodes can be related in radically different ways. `TENSIONS_WITH` and `OPPOSES` are not the same thing — a tension is a live contradiction you hold simultaneously (and probably can't resolve), while opposition is direct antagonism. `INFORMS` and `REQUIRES` are not the same — a thinker can inform your epistemology without being required by it. These distinctions matter for reasoning: Calliope reads them differently and responds differently.

**Low node density enables emergent synthesis.** Because almost anything can be a `Concept` or a `Value`, the graph has fewer isolated islands and more potential paths between distant nodes. When Calliope calls `find_path` between two seemingly unrelated nodes — say, between an aesthetic preference and a political commitment — it can often find a route. That path is not just an interesting fact. It's a candidate insight: something worth examining, verifying, or disputing. The low-density taxonomy keeps the graph traversable in ways that enable genuine synthesis.

### The alternative approach — and when it's right

The opposing design is a **dense, domain-specific ontology**: many fine-grained node types (30, 50, 100+), strict typing rules, tightly constrained edge semantics. Think of biomedical knowledge graphs, legal ontologies, or enterprise knowledge management systems.

That approach is better when you need:
- **Precision over fluidity.** If the distinction between a `DiagnosticCriterion` and a `SymptomCluster` matters for downstream reasoning, you need that granularity encoded explicitly.
- **Controlled traversal.** Tight typing means fewer spurious paths. The graph will not suggest that two things are related unless they are related in a specified way.
- **Auditability.** Compliance, legal, and medical systems often need to explain *why* a connection was made. A strict ontology makes those explanations tractable.
- **Integration with external standards.** If your graph needs to interoperate with established schemas (SNOMED, OWL, FHIR), you need to match their taxonomies.

The cost is exactly what Calliope sacrifices: fluidity, the possibility of unexpected synthesis, and the ability to encode genuinely novel concepts without first extending the schema. A dense ontology is excellent at representing what is already known within a domain. It is poor at representing the edges of someone's thinking — the places where a concept from one domain lights up an entirely different one.

Calliope is built for the edges.

---

## The graph file

The graph lives in `data/graph.json` — a flat JSON file with `nodes` and `edges` arrays. Each node has an `id`, `type`, `description`, and `weight` (credence: 0.0 = tentative, 1.0 = bedrock). Each edge has a `source`, `target`, `type`, `weight`, and `grounding` (a natural-language justification for the claim).

The repo ships with an **example graph** featuring a fictional interlocutor. Replace it with your own. The graph grows through conversation — Calliope will add to it as you talk.

---

## Current implementation

Calliope now combines a mutable knowledge graph with a searchable reading and writing corpus:

- **Hybrid retrieval:** Nomic embeddings through LM Studio plus a BM25 lexical index, reciprocal rank fusion, and an optional cross-encoder reranker. Relevance thresholds can be calibrated against the local corpus; search can return no relevant passages.
- **Source diversity:** a per-source result cap limits domination by heavily represented books. Source metadata distinguishes the owner's writing from published texts and saved model dialogue, with a preference for the owner's writing.
- **Corpus ingestion:** PDF and plain-text extraction, readability triage, optional local vision OCR, paragraph-aware chunks with overlap and page metadata, stable chunk identifiers, and incremental indexing in ChromaDB. Rebuild after changing chunking or embedding settings.
- **Semantic graph search:** cached node embeddings, exact-name preference, type filtering, duplicate candidates, and an explicit lexical fallback if embeddings are unavailable.
- **Provenance and grounding:** assertions distinguish `stated`, `confirmed`, `inferred`, and `derived`. Passage nodes link claims back to corpus chunks; the model's interpretation of a passage remains an inference.
- **Temporal tracking:** first-asserted dates, revision history, retirement records, and change queries. Older undated assertions retain a pre-temporal marker rather than an invented date.
- **Conversation controls:** save an exchange or only the user's words, inspect changes, and reset the conversation. Graph update reports are assembled from actual tool results.
- **Runtime controls:** configuration lives in `config.py`; the model client checks the server's context allocation, trims old tool results, and attempts to continue truncated answers.

The dated architecture illustration in `Calliope/calliope_architecture.html` is retained as an earlier reference. The modules and this README describe the current implementation.

---

## Setup

Requirements: Python 3.10+, [LM Studio](https://lmstudio.ai/) with its local server enabled, a chat model supporting tool calls, and Nomic Embed Text v1.5 for semantic search and corpus ingestion.

From the repository root:

```bash
python3 -m pip install -r requirements.txt
cd Calliope
python3 main.py
```

The defaults are `google/gemma-4-12b-qat` for chat, `text-embedding-nomic-embed-text-v1.5` for embeddings, and `http://localhost:1234/v1` for LM Studio. Use the identifiers served by your installation:

```bash
CALLIOPE_MODEL="your-loaded-model" python3 main.py
CALLIOPE_EMBED_MODEL="your-embedding-model" python3 main.py
CALLIOPE_CONTEXT=32651 CALLIOPE_DEBUG=1 python3 main.py
```

Set `CALLIOPE_CONTEXT` to match the context allocated in LM Studio. Other configuration includes `CALLIOPE_LM_URL`, `CALLIOPE_RERANK_MODEL`, and `CALLIOPE_ABS_BASELINE`. If the optional reranker cannot load, retrieval reports the fallback.

### Build your corpus

Create these directories under `Calliope/` as needed:

| Directory | Content |
|---|---|
| `corpus/` | Source PDFs |
| `corpus_text/` | Published works as `.txt` or `.md` |
| `corpus_sean/` | The owner's writing as `.txt` or `.md` (the directory name is currently fixed in code) |
| `corpus_dialogue/` | Saved user/model exchanges, indexed separately from original writing |

With LM Studio serving the embedding model, run from `Calliope/`:

```bash
python3 ingest.py --no-ocr
python3 ingest.py --report
python3 corpus.py --calibrate
python3 graph_search.py --calibrate
```

Omit `--no-ocr` to enable vision OCR for unreadable PDF pages; this also needs the vision model configured in `ingest.py`. `python3 ingest.py --rebuild` recreates the collection from the source files. Check `SOURCE_METADATA` in `ingest.py` when adding texts so citations carry accurate authors and titles.

The public repository contains an example graph. Personal graphs, source texts, saved conversations, indexes, caches, and calibration files are not included in this update. Keep your personal graph outside Git tracking: `.gitignore` cannot hide changes to the already tracked example at `Calliope/data/graph.json`.

### Terminal commands

| Command | Action |
|---|---|
| `graph` | Inspect graph nodes and counts |
| `corpus <query>` | Search the corpus with retrieval diagnostics |
| `keep` | Save the latest complete exchange to the dialogue corpus |
| `keep me` | Save only the user's latest turn to the owner's writing corpus |
| `changed` / `changed since YYYY-MM-DD` | Inspect graph changes |
| `reset` | Clear conversation history |
| `exit` | Quit |

Run ingestion again after saving exchanges to make them searchable.

### Review graph weights

From `Calliope/`, use `python3 weights.py --export`, edit the generated `weights_todo.yaml`, then run `python3 weights.py --apply`. `python3 weights.py --status` reports completion.

### Verification

These tests run without LM Studio; semantic search uses deterministic embedding stubs:

```bash
cd Calliope
python3 test_graph_nodes.py
python3 test_graph_temporal.py
python3 test_graph_search.py
```

They cover node safeguards, history and provenance, passage grounding, graph search ranking and filtering, cache invalidation, and lexical fallback. They do not establish live model or retrieval quality.

---

## Project structure

| File | Responsibility |
|---|---|
| `Calliope/main.py` | CLI, tool loop, conversation commands |
| `Calliope/config.py` | Model endpoint, context and tool budgets |
| `Calliope/model_io.py` | Model calls, streaming, continuation, context checks |
| `Calliope/graph_report.py` | Reports verified graph changes |
| `Calliope/loop.py` | System prompt and protocol requirements |
| `Calliope/tools.py` | Tool schemas and dispatch |
| `Calliope/graph.py` | NetworkX graph, persistence, provenance, history |
| `Calliope/graph_search.py` | Semantic graph search and embedding cache |
| `Calliope/embeddings.py` | Shared embedding client and task prefixes |
| `Calliope/corpus.py` | Hybrid corpus retrieval and calibration |
| `Calliope/ingest.py` | PDF/text extraction, chunking, indexing |
| `Calliope/weights.py` | Export and apply weight reviews |
| `Calliope/data/graph.json` | Public example graph; replace locally for personal use |

---

## The name

Calliope is the muse of epic poetry in Greek mythology — the patron of long-form, serious, sustained work. The name is a reminder of what this project is for: not quick answers, but the slow development of a worldview. Muses do not flatter. They press.

---

## License

MIT


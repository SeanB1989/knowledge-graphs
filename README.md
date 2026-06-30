[README.md](https://github.com/user-attachments/files/29525429/README.md)
# Calliope

**A personal ontology engine. A different kind of AI companion.**

Calliope is a local AI that knows what you care about — your values, your goals, the tensions you live inside, the thinkers who shaped you. Before it says anything, it reads the graph. It notices when you're drifting. It helps you figure out what to do about that.

It runs entirely on your machine, on an open-weights model via [LM Studio](https://lmstudio.ai/). No cloud. No subscriptions. No data leaving your computer.

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

Calliope uses **8 node types**:

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

This is a deliberately **small taxonomy**. Almost any object — a fear, a philosophy, a habit, a relationship — can be encoded within it, but it must be classified into one of these eight categories. That constraint is doing work.

`Belief` absorbs what other systems call *epistemic* nodes — stances toward knowing are beliefs, second-order ones. `Value` absorbs aesthetic commitments — aesthetic preferences function as values in the graph: they carry weight, guide decisions, and can tension against other things. `Concept` absorbs self-positioning models — how you understand your own role in your work and world is part of the conceptual vocabulary you think with. The consolidation isn't loss; it's clarity about what these things actually are.

### The edge vocabulary

The graph has **10 edge types**:

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

## Next steps: corpus, vectors, and grounded edges

The current graph is an assertion engine: Calliope asserts new nodes and edges based on reasoning, with weights and groundings supplied as natural language. This works, but it has a limit. The grounding is still Calliope's inference — it has no access to primary sources.

The next significant development is a **grounded epistemological layer**:

1. **Corpus ingestion.** Feed Calliope your actual writing — journals, essays, notes — alongside the texts that have shaped you: philosophy, literature, science, criticism, and media (it can process images). This corpus is the raw material of your actual intellectual life.

2. **Vectorisation and RAG.** Chunk and embed the corpus into a vector database. When Calliope asserts an edge, it queries the corpus for passages that support the claim — retrieving the actual text (or image, or source) that grounds the relationship.

3. **Reified edge grounding.** Instead of `grounding: "Sean has said he finds restraint aesthetically necessary"`, an edge could carry `grounding: "Journal entry, March 2024: 'The version without the third paragraph is the one.'"` — a direct citation from primary material.

This changes the epistemological status of the graph significantly. Edges stop being Calliope's inferences about you and become claims anchored in your actual expressed thought. The graph becomes not just a model of your worldview but an **evidence-based record** of it — one that can be interrogated, audited, and revised with reference to the source material.

It also enables a different kind of conversation. When Calliope finds a tension, it could surface the passages from your own writing where that tension shows up — not as illustration but as evidence. The conversation becomes a dialogue with your own archive.

---

## Setup

**Requirements:**
- Python 3.9+
- [LM Studio](https://lmstudio.ai/) running locally with the server enabled
- A model loaded in LM Studio (default: `google/gemma-4-12b-qat`)

```bash
pip install openai networkx
```

**Configure** `main.py` if your model name differs:
```python
MODEL = "google/gemma-4-12b-qat"  # change to whatever you have loaded
LM_STUDIO_URL = "http://localhost:1234/v1"
```

**Run:**
```bash
python3 main.py
```

**Commands in the REPL:**
- Type anything to talk to Calliope
- `graph` — inspect the current graph (node counts by type)
- `exit` — quit

---

## Project structure

```
calliope/
├── main.py          # Entry point, ReAct loop, CLI REPL
├── graph.py         # CalliopeGraph class — all graph operations
├── tools.py         # Tool definitions (OpenAI function calling format) + execution
├── loop.py          # System prompt, continuation logic, REQUIRED_TOOLS enforcement
└── data/
    └── graph.json   # The ontology graph (replace with your own)
```

---

## The name

Calliope is the muse of epic poetry in Greek mythology — the patron of long-form, serious, sustained work. The name is a reminder of what this project is for: not quick answers, but the slow development of a worldview. Muses do not flatter. They press.

---

## License

MIT

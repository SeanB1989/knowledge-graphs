"""
Calliope Corpus Ingestion (v2)
==============================

Builds the searchable index Calliope reads from. Run it once, and again
whenever you add texts.

    python3 ingest.py                 # incremental: only new/changed files
    python3 ingest.py --rebuild       # wipe and rebuild from scratch
    python3 ingest.py --no-ocr        # skip slow vision OCR; drop bad pages
    python3 ingest.py --report        # just print what's in the index, change nothing

WHAT THIS FILE DOES, IN ORDER
-----------------------------
  1. EXTRACT  -- pull text out of each source file, page by page.
  2. TRIAGE   -- score each page's readability. A scanned PDF often has a
                 *corrupt* text layer rather than none at all, so "did we get
                 any text?" is the wrong question. We ask "is this text
                 actually language?"
  3. OCR      -- pages that fail triage get re-read by the local vision model.
  4. CHUNK    -- split into passages on paragraph boundaries, tracking which
                 pages each passage spans so citations can be precise.
  5. FILTER   -- drop any chunk that is still unreadable after all that.
  6. EMBED    -- turn each chunk into a vector (see embeddings.py).
  7. STORE    -- write vectors + text + metadata into ChromaDB.

WHAT WENT WRONG IN v1 (and why this rewrite exists)
---------------------------------------------------
  * The Kaczynski PDF is a scan whose text layer decodes to noise -- things
    like "5 § i f * 5 =a 5* - » . %". v1 checked `if text and text.strip()`,
    which was TRUE for noise, so the OCR fallback never fired. 174 of its 179
    chunks were pure garbage: 16% of the entire corpus was anti-signal, and
    because those chunks still got embedded, they could and did outrank real
    passages on short or unusual queries.
  * The Zizek PDF is 8 scanned pages of a completely different book (a
    Blackwell literary theory anthology). It produced exactly ONE usable chunk.
    Calliope has been citing Zizek from its own training data, not from the
    corpus, because there was nothing there to retrieve.
  * Chunks were fixed 400-word windows cut mid-sentence, with no page numbers,
    so nothing could be cited by location.
  * The embedding model truncated every chunk at ~256 tokens (~45%).

The triage step below is the fix for the first two. It is deliberately a dumb,
transparent heuristic rather than anything clever -- see READABILITY below.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

try:
    import pdfplumber
except ImportError:
    print("Missing: pip3 install pdfplumber")
    sys.exit(1)

try:
    import chromadb
except ImportError:
    print("Missing: pip3 install chromadb")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embeddings

# -- Config -------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")            # PDFs
CORPUS_TEXT_DIR = os.path.join(HERE, "corpus_text")  # clean .txt/.md of published works
CORPUS_SEAN_DIR = os.path.join(HERE, "corpus_sean")  # Sean's own writing
# Saved exchanges with Calliope. Kept SEPARATE from corpus_sean on purpose:
# Sean's turns are his words, but Calliope's are model output, and letting model
# output re-enter the corpus tagged as Sean's thinking is precisely the failure
# his own graph names "simulated knowledge" -- the counterfeit of genuine
# expertise, indistinguishable in texture. The `keep me` command writes his turn
# alone to corpus_sean; plain `keep` writes the whole exchange here.
CORPUS_DIALOGUE_DIR = os.path.join(HERE, "corpus_dialogue")
CHROMA_DIR = os.path.join(HERE, "data", "chroma")

# New collection: the vectors are 768-dim Nomic now, not 384-dim MiniLM.
# Chroma cannot mix dimensions in one collection, and even if it could, the
# distances would be meaningless across models.
COLLECTION_NAME = (
    f"calliope_corpus_{embeddings.EMBED_SPACE_VERSION}".replace(".", "_").replace("-", "_")
)

TARGET_CHUNK_WORDS = 320   # what we aim for
MAX_CHUNK_WORDS = 450      # a single paragraph longer than this gets sentence-split
OVERLAP_WORDS = 60         # carry-over so ideas straddling a boundary survive

# Note these are targets, not hard caps. A chunk is the carried-over overlap
# plus one paragraph, so the real worst case is MAX_CHUNK_WORDS + OVERLAP_WORDS
# = 510 words, about 690 tokens. That is deliberate: paragraph integrity is
# worth more than exact uniformity, and Nomic's window is 8192 tokens, so there
# is an order of magnitude of headroom. It would NOT have been safe under the
# old MiniLM model, whose 256-token limit is what caused the silent truncation
# this rewrite exists to fix.

VISION_MODEL = os.environ.get("CALLIOPE_VISION_MODEL", "google/gemma-4-12b-qat")
LM_STUDIO_URL = embeddings.LM_STUDIO_URL


# -- READABILITY --------------------------------------------------------------
# The whole triage step rests on one observation: real English prose is mostly
# made of words that are three or more letters long. Mangled OCR output is
# mostly made of stray single characters, punctuation and digits.
#
# Measured across the existing index, the separation is not subtle:
#
#     Capitalist Realism (clean text layer)   0.67 - 0.83
#     What Is History?   (clean text layer)   0.66 - 0.82
#     The Nuer           (clean text layer)   0.70 - 0.86   (a few bad pages)
#     Sean's own essays                       0.60 - 0.94
#     Kaczynski          (corrupt scan)       0.00 - 0.04
#
# There is a gap of half the scale between the worst real text and the best
# garbage, so any threshold in 0.3-0.6 works. 0.55 leaves plenty of room while
# still comfortably clearing every genuine source.

WORDLIKE = re.compile(r"^[A-Za-z][A-Za-z'’-]{2,}$")

READABLE_PAGE = 0.55     # below this, a page gets sent to vision OCR
READABLE_CHUNK = 0.50    # below this, a chunk is dropped entirely
MIN_WORDS_TO_JUDGE = 12  # too short to score meaningfully; let it through


def readability(text: str) -> float:
    """
    Fraction of whitespace-separated tokens that look like real words.
    Returns 0.0-1.0. Higher is more language-like.
    """
    words = text.split()
    if len(words) < MIN_WORDS_TO_JUDGE:
        return 1.0  # don't judge fragments; headers and captions are legitimate
    hits = sum(1 for w in words if WORDLIKE.match(w.strip(".,;:!?()\"[]{}<>")))
    return hits / len(words)


# -- Source metadata ----------------------------------------------------------
# Maps a filename stem to (author, title). Anything not listed here falls back
# to a humanised version of the filename, and is reported at the end so you
# notice rather than silently shipping "Unknown".

SOURCE_METADATA = {
    "capitalist_realism_fisher":  ("Mark Fisher", "Capitalist Realism"),
    "sublime_object_zizek":       ("Slavoj Žižek", "The Sublime Object of Ideology"),
    "what_is_history_carr":       ("E.H. Carr", "What Is History?"),
    "unabomber_manifesto":        ("Ted Kaczynski", "Industrial Society and Its Future"),
    "the_nuer_evans_pritchard":   ("E.E. Evans-Pritchard", "The Nuer"),
    "Evans_Pritchard_E_E_The_Nuer_a_description_of_the_modes_1940":
                                  ("E.E. Evans-Pritchard", "The Nuer"),

    "sean_frameworks_dont_exist":    ("Sean Bruce", "Frameworks Don't Exist to Improve Decisions"),
    "sean_jerry_mayor_of_nowhere":   ("Sean Bruce", "Jerry, Mayor of Nowhere"),
    "sean_hell_in_san_miguel":       ("Sean Bruce", "Hell in San Miguel"),
    "sean_nameless_dread":           ("Sean Bruce", "Nameless Dread"),
    "sean_all_these_foreign_things": ("Sean Bruce", "All These Foreign Things"),
    "sean_emily_vignette":           ("Sean Bruce", "Emily (A Vignette)"),
    "sean_philly_man":               ("Sean Bruce", "Philly Man"),
    "sean_smooth_landing":           ("Sean Bruce", "Smooth Landing"),
    "sean_behold_manifesto":         ("Sean Bruce", "BEHOLD!"),
    "sean_holy_filth":               ("Sean Bruce", "Holy Filth"),
    "sean_i_hate_it_here":           ("Sean Bruce", "I Hate It Here"),
    "sean_insufferable_city":        ("Sean Bruce", "Insufferable City"),
    "sean_the_remainder":            ("Sean Bruce", "The Remainder"),
    "sean_the_alibi_machine":        ("Sean Bruce", "The Alibi Machine"),
    "sean_working_constitution":     ("Sean Bruce", "A Working Constitution of My Worldview"),
    "sean_vertigo":                  ("Sean Bruce", "Vertigo"),

    "veblen_theory_of_the_leisure_class": ("Thorstein Veblen", "The Theory of the Leisure Class"),
    "marx_engels_communist_manifesto":    ("Karl Marx and Friedrich Engels", "The Communist Manifesto"),
    "nietzsche_genealogy_of_morals":      ("Friedrich Nietzsche", "On the Genealogy of Morals"),
    "dostoevsky_notes_from_underground":  ("Fyodor Dostoevsky", "Notes from Underground"),
    "kafka_the_metamorphosis":            ("Franz Kafka", "The Metamorphosis"),
    "marcus_aurelius_meditations":        ("Marcus Aurelius", "Meditations"),
    "wilde_the_picture_of_dorian_gray":   ("Oscar Wilde", "The Picture of Dorian Gray"),
    "thoreau_civil_disobedience":         ("Henry David Thoreau", "On the Duty of Civil Disobedience"),
}


def lookup_metadata(stem: str) -> tuple[str, str, bool]:
    """Return (author, title, was_known)."""
    if stem in SOURCE_METADATA:
        a, t = SOURCE_METADATA[stem]
        return a, t, True
    pretty = stem.replace("_", " ").replace("-", " ").strip().title()
    return "Unknown", pretty, False


# -- Vision OCR ---------------------------------------------------------------

_vlm = None


def _get_vlm():
    global _vlm
    if _vlm is None:
        from openai import OpenAI
        _vlm = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    return _vlm


def ocr_page(page) -> str:
    """
    Render a page to an image and have the local vision model transcribe it.

    Note this is genuinely slow -- seconds to tens of seconds per page on a
    laptop. That is why triage matters: we only pay this cost for pages that
    actually need it, rather than OCR-ing whole books.
    """
    import base64
    from io import BytesIO

    try:
        img = page.to_image(resolution=200).original
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        resp = _get_vlm().chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Transcribe all text on this page exactly as it appears, verbatim. "
                        "Preserve paragraph breaks. Read multi-column layouts in reading "
                        "order, one column fully before the next. Do not summarize, "
                        "comment, translate, or add anything not on the page. "
                        "If there is no legible text, reply with nothing."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            temperature=0.0,
            max_tokens=2400,
        )
        out = resp.choices[0].message.content or ""
        # The VLM sometimes editorialises despite instructions. If its output
        # is itself unreadable, treat the page as a loss rather than poison
        # the index with hallucinated filler.
        return out if readability(out) >= READABLE_PAGE else ""
    except Exception as e:
        print(f"\n      OCR error: {e}")
        return ""


# -- Extraction ---------------------------------------------------------------
# Everything returns list[(page_number, text)] so page provenance survives all
# the way into the stored metadata.

def extract_pdf(path: str, use_ocr: bool = True) -> tuple[list[tuple[int, str]], dict]:
    pages: list[tuple[int, str]] = []
    stats = {"total": 0, "clean": 0, "ocr_ok": 0, "ocr_failed": 0, "dropped": 0}

    with pdfplumber.open(path) as pdf:
        stats["total"] = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            score = readability(raw) if raw.strip() else 0.0

            if raw.strip() and score >= READABLE_PAGE:
                pages.append((i, raw))
                stats["clean"] += 1
                continue

            # Either no text layer, or -- the case v1 missed entirely -- a text
            # layer that exists but decodes to noise.
            if not use_ocr:
                stats["dropped"] += 1
                continue

            label = "no text layer" if not raw.strip() else f"unreadable text layer ({score:.2f})"
            print(f"    page {i}/{stats['total']}: {label}, running vision OCR...",
                  end="", flush=True)
            text = ocr_page(page)
            if text.strip():
                pages.append((i, text))
                stats["ocr_ok"] += 1
                print(" ok")
            else:
                stats["ocr_failed"] += 1
                print(" failed, dropping page")

    return pages, stats


def extract_plaintext(path: str, use_ocr: bool = True) -> tuple[list[tuple[int, str]], dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [(1, text)], {"total": 1, "clean": 1, "ocr_ok": 0, "ocr_failed": 0, "dropped": 0}


# -- Chunking -----------------------------------------------------------------

def _paragraphs(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Flatten pages into (page_number, paragraph) blocks."""
    blocks = []
    for page_no, text in pages:
        text = re.sub(r"-\n(\w)", r"\1", text)                    # de-hyphenate line breaks
        text = re.sub(r"[ \t]+\n", "\n", text)                     # drop trailing spaces
        # Unwrap soft line breaks. The \n inside the lookbehind is load-bearing:
        # without it, the SECOND newline of a blank-line paragraph break matched
        # (its preceding char is a newline, which was not in the protected set)
        # and was replaced with a space. "\n\n" became "\n ", so the paragraph
        # splitter below -- which looks for two newlines -- could never fire, and
        # every page collapsed into a single paragraph. Only 3+ consecutive
        # newlines survived. Measured effect before the fix: 44% of chunks landed
        # in a 300-330 word band, the signature of the sentence-split fallback
        # doing all the work that paragraph packing was supposed to do.
        text = re.sub(r"(?<![.!?:;\"'\)\n])\n(?!\n)", " ", text)  # unwrap soft line breaks
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if para:
                blocks.append((page_no, para))
    return blocks


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """
    Pack paragraphs into chunks of ~TARGET_CHUNK_WORDS, never splitting a
    paragraph unless it alone exceeds MAX_CHUNK_WORDS.

    Why paragraphs and not fixed windows: an embedding represents ONE point in
    meaning-space. A chunk that starts halfway through an argument about
    ideology and ends halfway into a different one about labour produces a
    vector sitting between the two, and is a good match for neither. Aligning
    to paragraph boundaries keeps each vector about one thing.

    Each chunk records the page range it covers, so Calliope can cite
    "Fisher, Capitalist Realism, pp. 12-13" instead of gesturing vaguely.
    """
    blocks = _paragraphs(pages)
    chunks: list[dict] = []

    cur: list[str] = []
    cur_words = 0
    cur_pages: list[int] = []

    def flush():
        nonlocal cur, cur_words, cur_pages
        if not cur or not cur_pages:
            return
        text = " ".join(cur)
        chunks.append({
            "text": text,
            "page_start": min(cur_pages),
            "page_end": max(cur_pages),
        })
        # Overlap: carry the tail of this chunk into the next one so a claim
        # that straddles the boundary is still retrievable from either side.
        words = text.split()
        tail = words[-OVERLAP_WORDS:] if OVERLAP_WORDS else []
        last_page = max(cur_pages)
        cur = [" ".join(tail)] if tail else []
        cur_words = len(tail)
        cur_pages = [last_page] if tail else []

    for page_no, para in blocks:
        pw = len(para.split())

        # A single paragraph too big to fit: hard-split it on sentences.
        if pw > MAX_CHUNK_WORDS:
            flush()
            cur, cur_words, cur_pages = [], 0, []
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf, bw = [], 0
            for s in sentences:
                sw = len(s.split())
                if bw + sw > TARGET_CHUNK_WORDS and buf:
                    chunks.append({"text": " ".join(buf),
                                   "page_start": page_no, "page_end": page_no})
                    buf, bw = [], 0
                buf.append(s)
                bw += sw
            if buf:
                cur = [" ".join(buf)]
                cur_words = bw
                cur_pages = [page_no]
            continue

        if cur_words + pw > TARGET_CHUNK_WORDS and cur:
            flush()

        cur.append(para)
        cur_words += pw
        cur_pages.append(page_no)

    # Don't emit a final chunk that is nothing but leftover overlap text.
    if cur_words >= 20:
        flush()

    return chunks


# -- File discovery -----------------------------------------------------------

SOURCE_DIRS = [
    (CORPUS_DIR,      (".pdf",),       extract_pdf,       "published"),
    (CORPUS_TEXT_DIR, (".txt", ".md"), extract_plaintext, "published"),
    (CORPUS_SEAN_DIR, (".txt", ".md"), extract_plaintext, "sean"),
    (CORPUS_DIALOGUE_DIR, (".txt", ".md"), extract_plaintext, "dialogue"),
]


def file_fingerprint(path: str) -> str:
    """Hash of file contents -- lets us skip files that haven't changed."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -- Main ---------------------------------------------------------------------

def get_collection(client, rebuild: bool):
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def report(collection):
    import collections as _c
    data = collection.get(include=["metadatas", "documents"])
    metas = data["metadatas"] or []
    docs = data["documents"] or []
    if not metas:
        print("Index is empty.")
        return

    by_title = _c.defaultdict(list)
    for m, d in zip(metas, docs):
        by_title[m.get("title", "?")].append(readability(d))

    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Total chunks: {len(metas)}\n")
    print(f"{'title':<48} {'chunks':>7} {'readability':>12}")
    print("-" * 70)
    for t, scores in sorted(by_title.items(), key=lambda kv: -len(kv[1])):
        mean = sum(scores) / len(scores)
        flag = "  <-- SUSPECT" if mean < READABLE_CHUNK else ""
        print(f"{t[:48]:<48} {len(scores):>7} {mean:>12.2f}{flag}")


def ingest(rebuild: bool = False, use_ocr: bool = True):
    print("\nCalliope Corpus Ingestion (v2)")
    print("=" * 60)
    print(f"Index      : {CHROMA_DIR}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Embeddings : {embeddings.EMBED_MODEL} via {LM_STUDIO_URL}")
    print(f"Vision OCR : {'enabled (' + VISION_MODEL + ')' if use_ocr else 'DISABLED'}")
    print()

    # Fail fast and loudly if LM Studio isn't reachable, rather than after an
    # hour of OCR.
    try:
        dim = embeddings.dimension()
        print(f"Embedding endpoint OK ({dim} dimensions).\n")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = get_collection(client, rebuild)

    existing = collection.get(include=["metadatas"])
    existing_ids = set(existing["ids"])
    seen_fingerprints = {m.get("file_hash") for m in (existing["metadatas"] or [])}

    total_added = 0
    unknown_sources: list[str] = []
    quality_rows: list[tuple] = []

    for directory, exts, extractor, provenance in SOURCE_DIRS:
        if not os.path.isdir(directory):
            continue
        files = sorted(f for f in os.listdir(directory)
                       if f.lower().endswith(exts) and not f.startswith("."))
        for filename in files:
            path = os.path.join(directory, filename)
            stem = os.path.splitext(filename)[0]
            author, title, known = lookup_metadata(stem)
            if not known:
                unknown_sources.append(stem)

            fhash = file_fingerprint(path)
            if fhash in seen_fingerprints and not rebuild:
                print(f"Skipping (unchanged): {title}")
                continue

            print(f"\nProcessing: {title} -- {author}")
            try:
                pages, stats = extractor(path, use_ocr=use_ocr)
            except Exception as e:
                print(f"  ERROR extracting: {e}")
                continue

            if not pages:
                print("  WARNING: no usable text extracted. Skipping.")
                quality_rows.append((title, 0, 0.0, stats))
                continue

            chunks = chunk_pages(pages)

            kept, dropped = [], 0
            for ch in chunks:
                if readability(ch["text"]) < READABLE_CHUNK:
                    dropped += 1
                    continue
                kept.append(ch)

            if stats["total"] > 1:
                print(f"  pages: {stats['clean']} clean, {stats['ocr_ok']} OCR'd, "
                      f"{stats['ocr_failed']} OCR failed, {stats['dropped']} dropped")
            print(f"  chunks: {len(kept)} kept, {dropped} dropped as unreadable")

            if not kept:
                print("  WARNING: every chunk was unreadable. Nothing indexed.")
                quality_rows.append((title, 0, 0.0, stats))
                continue

            mean_read = sum(readability(c["text"]) for c in kept) / len(kept)
            quality_rows.append((title, len(kept), mean_read, stats))

            ids, docs, metas = [], [], []
            for i, ch in enumerate(kept):
                cid = hashlib.md5(f"{stem}|{i}|{ch['text'][:200]}".encode()).hexdigest()
                if cid in existing_ids:
                    continue
                ids.append(cid)
                docs.append(ch["text"])
                metas.append({
                    "source": stem,
                    "author": author,
                    "title": title,
                    "provenance": provenance,   # "sean" vs "published"
                    "page_start": ch["page_start"],
                    "page_end": ch["page_end"],
                    "chunk_index": i,
                    "total_chunks": len(kept),
                    "file_hash": fhash,
                    "readability": round(readability(ch["text"]), 3),
                })

            if not ids:
                print("  Already indexed.")
                continue

            vectors = embeddings.embed(docs, kind="document", progress=True)
            collection.add(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
            existing_ids.update(ids)
            total_added += len(ids)

    print("\n" + "=" * 60)
    print(f"Ingestion complete. {total_added} new chunks. Index total: {collection.count()}")

    if quality_rows:
        print("\nQuality report")
        print(f"{'title':<44} {'chunks':>7} {'readability':>12}")
        print("-" * 66)
        for title, n, r, _ in sorted(quality_rows, key=lambda x: x[2]):
            flag = ""
            if n == 0:
                flag = "  <-- NOTHING INDEXED"
            elif r < 0.6:
                flag = "  <-- CHECK THIS"
            print(f"{title[:44]:<44} {n:>7} {r:>12.2f}{flag}")

    if unknown_sources:
        print("\nSources with no entry in SOURCE_METADATA (citations will be weak):")
        for s in unknown_sources:
            print(f"  - {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build Calliope's corpus index.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete the collection and rebuild from scratch.")
    ap.add_argument("--no-ocr", action="store_true",
                    help="Skip vision OCR; drop unreadable pages instead.")
    ap.add_argument("--report", action="store_true",
                    help="Print index contents and exit without changing anything.")
    args = ap.parse_args()

    if args.report:
        if not os.path.exists(CHROMA_DIR):
            print(f"No index directory at {CHROMA_DIR}.\n"
                  f"Build one first:  python3 ingest.py --rebuild")
            sys.exit(1)

        c = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            report(c.get_collection(COLLECTION_NAME))
        except Exception:
            # --report only reads. It never builds anything, so "does not
            # exist" almost always means the rebuild hasn't been run yet.
            others = []
            try:
                others = [col.name for col in c.list_collections()]
            except Exception:
                pass

            print(f"\nNo index yet for the current embedding model.")
            print(f"  Expected collection: {COLLECTION_NAME}")
            if others:
                print(f"  Collections present: {', '.join(others)}")
                if any(o != COLLECTION_NAME for o in others):
                    print("\n  The collection(s) above were built with the old "
                          "384-dimension MiniLM\n  embeddings. They cannot be read "
                          "with the new 768-dimension model,\n  and are left in "
                          "place only so nothing is destroyed.")
            else:
                print("  Collections present: none")

            print("\nRun this first:")
            print("    python3 ingest.py --rebuild")
            print("\n(--report only reads an existing index; it never builds one.)")
            sys.exit(1)
        sys.exit(0)

    ingest(rebuild=args.rebuild, use_ocr=not args.no_ocr)

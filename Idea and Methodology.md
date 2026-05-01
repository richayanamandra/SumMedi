# SumMedi — Full Analysis

## 🗺️ Project Overview

**SumMedi** is a Python 3.12 implementation and extention of the academic paper *"Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation"* (Wu et al., 2024).

Its central innovation is a **triple-layer knowledge graph** backed by Neo4j, combined with a hierarchical **U-Retrieval engine** to produce evidence-based, hallucination-resistant medical answers. The front-end is a Streamlit UI.

---

## 🏗️ Architecture: The Three Layers

<img width="2816" height="1363" alt="architecture" src="https://github.com/user-attachments/assets/3704d01d-d324-4dd3-b3d1-e0d7e998c26a" />


| Layer | Role | Data Source | Storage | Python RAM |
|-------|------|-------------|---------|-----------|
| **Layer 1** (RAG Graph) | Patient/User documents | EHR, clinical notes, paste text, URLs | Neo4j (synced) | ✅ `meta_graphs` list |
| **Layer 2** (Repository Graph) | Medical literature | PubMed abstracts, MeSH terms | Neo4j (persistent) | ✅ `repo_entities_l2` list |
| **Layer 3** (Vocabulary Graph) | UMLS ontology | UMLS dump (MRCONSO.RRF), ~3.5M nodes | **Neo4j only** | ❌ Zero-RAM design |

**Cross-layer edges:**
- `the_reference_of`: L1 entity → L2 entity (cosine similarity ≥ 0.45)
- `the_definition_of`: L2 entity → L3 entity (GPU-accelerated, threshold 0.60)

---

## 📁 File-by-File Breakdown

### `app.py` — Streamlit Frontend (693 lines)
The entire UI. Three tabs:
1. **Query tab** — Run U-Retrieval, shows answer + retrieved entities + bottom-up refinement log
2. **Graph Explorer tab** — Interactive PyVis network OR entity table view with tag tree
3. **Architecture tab** — Static explainer of the system

**Key behaviours:**
- Initialises `MedGraphRAG` in `st.session_state` (persists across reruns)
- Sidebar handles document loading (URL via `WebBaseLoader`, or direct paste)
- "Clear Patient Documents (Layer 1)" button: wipes only L1 from both RAM and Neo4j
- Neo4j connection status badge shown in sidebar
- Custom dark-mode CSS (IBM Plex fonts, GitHub dark palette)

---

### `med_graph_rag.py` — Core Engine (875 lines)

**Class:** `MedGraphRAG`

**Constants (tunable thresholds):**
```python
SIMILARITY_THRESHOLD = 0.45   # δr — L1↔L2 linking threshold
TAG_MERGE_THRESHOLD  = 0.60   # δt — hierarchical tag clustering
TOP_N_ENTITIES       = 8      # Nu — entities retrieved per query
TOP_K_NEIGHBOURS     = 2      # ku — triple-neighbour hops
MAX_TRIPLE_NEIGHBORS = 50     # safety cap for context window
MAX_TAG_LAYERS       = 6      # max U-Retrieval depth
```

**Graph Construction Pipeline (`load_documents`):**
1. **Semantic Chunking** — `RecursiveCharacterTextSplitter` (chunk_size=1200, overlap=150)
2. **Entity Extraction** — LLM → JSON array of `{name, type, context}` per chunk (parallel, 5 workers)
3. **Relationship Extraction** — LLM → JSON array of `{source, relation, target}` per chunk (parallel)
4. **NetworkX + Neo4j Sync** — Entities and relationships stored in both an in-memory `nx.DiGraph` and Neo4j
5. **Triple Linking (`_link_layers`)** — Vectorized NumPy cosine similarity to draw L1→L2 edges
6. **Tagging (`_tag_all_graphs`)** — LLM summarises each MetaMedGraph into predefined tag categories (parallel)
7. **Tag Tree (`_build_tag_tree`)** — O(N²) agglomerative clustering over tag embedding centroids

**U-Retrieval Pipeline (`query`):**
1. **Top-Down**: Traverse tag_tree depth-first, find leaf MetaMedGraph with max embedding similarity to query
2. **Top-N Entities**: Score entities in target graph by cosine similarity to query, take top 8
3. **Triple Neighbours**: For each top entity, Neo4j k-hop traversal (k=2) up to 50 total neighbours
4. **Initial Answer**: LLM prompt with entities + relationships + neighbours
5. **Bottom-Up Refinement**: Climb the tag tree; re-prompt LLM at each ancestor level to refine

**Other notable methods:**
- `seed_pubmed_literature()` — fetch on-demand PubMed abstracts for a query string
- `bulk_import_pubmed()` — delegates to `pubmed_importer.bulk_fetch_pubmed`
- `import_local_umls_dump()` / `import_local_umls_relationships_dump()` — load UMLS from raw RRF files
- `link_cross_layers_gpu()` — delegates to `cross_layer_linker.link_layers_gpu`
- `simulate_massive_vocab()` — stress-test with 500k synthetic L3 nodes
- `clear_layer1()` / `clear_all()` / `clear_all_relationships()` — fine-grained cleanup

---

### `data_models.py` — Dataclasses (90 lines)

Three core dataclasses:

```python
@dataclass
class Entity:
    name: str
    entity_type: str          # UMLS semantic type (one of 21 types)
    context: str              # 1-2 sentence description from LLM
    definition: str = ""      # Layer 3 only — formal definition
    layer: int = 1            # 1, 2, or 3
    embedding: np.ndarray     # 384-dim float32 (all-MiniLM-L6-v2)

@dataclass
class Relationship:
    source: str; relation: str; target: str

@dataclass
class MetaMedGraph:
    graph_id: str             # e.g. "chunk_0"
    entities: list[Entity]
    relationships: list[Relationship]
    tag_summary: dict[str, str]   # e.g. {"DIAGNOSIS": "...", "MEDICATION": "..."}
    source_text: str
```

**Constants defined here:**
- `UMLS_SEMANTIC_TYPES` — 21 allowed entity types
- `MEDICAL_TAGS` — 10 tag categories (`SYMPTOMS`, `MEDICATION`, `DIAGNOSIS`, etc.)
- `BUILTIN_VOCAB` — 15 hand-coded L3 entries as a fallback when UMLS API is unavailable

---

### `llm_helpers.py` — LLM + Embedding Utilities (230 lines)

**`EmbeddingStore`** — Wraps any HuggingFace/OpenAI embedder with an in-memory cache:
- `embed(text)` — single embed with cache
- `embed_batch(texts)` — calls `embed_documents` if available for speed
- `similarity(a, b)` — 1 - cosine distance

**LLM-calling functions (all call `gpt-4o-mini`):**
| Function | What it does |
|----------|-------------|
| `_call_llm_json(llm, prompt)` | Calls LLM, strips markdown fences, parses JSON |
| `_extract_entities(llm, chunk)` | Returns `list[Entity]` from raw text |
| `_extract_relationships(llm, chunk, entities)` | Returns `list[Relationship]` |
| `_tag_graph(llm, graph)` | Returns `dict[tag → description]` for a MetaMedGraph |
| `_generate_answer(llm, q, graph, entities, neighbours)` | Initial bottom-level answer |
| `_refine_answer(llm, q, prev_answer, summary)` | Bottom-up refinement step |

---

### `api_clients.py` — External Service Clients (397 lines)

**`UMLSClient`** — REST client for `uts-ws.nlm.nih.gov`:
- `get_term_details(term)` → `{name, definition, type, source}` — tries "exact" search, falls back to "words"
- In-memory cache to avoid duplicate API calls
- Fetches from NCI or MSH sources preferentially for quality definitions

**`Neo4jClient`** — All database interactions:
- `sync_entities()` — batched MERGE (1000 at a time) with `embedding` as float list
- `sync_relationships()` — batched MERGE of `RELATED_TO` edges
- `batch_add_cross_layer_edges()` — batched MERGE of `LINK` edges with similarity scores
- `get_k_hop_neighbors(name, k)` — Cypher wildcard traversal `[*1..k]`, returns up to 50
- `load_layer2_entities()` — restore L2 from DB on app startup (zero re-embedding needed)
- `clear_layer1_db()` / `clear_all_db()` / `clear_all_relationships()` — maintenance
- `count_layer3()` — fast COUNT query, no RAM loading
- `sync_tag_tree()` — persists hierarchical tag tree as `TagNode` nodes with `HAS_CHILD`/`INDEXES` edges
- `ensure_indexes()` — background-thread unique constraint on `Entity.name`

**`PubMedClient`** — NCBI E-utilities:
- `fetch_abstracts(query, max_results)` — esearch → efetch → split by `\n\n\n`

---

### `cross_layer_linker.py` — GPU-Accelerated L2→L3 Linker (214 lines)

A standalone script / importable module that runs **outside** the normal ingestion pipeline. Called via `med_graph_rag.link_cross_layers_gpu()`.

**Pipeline:**
1. Load all L2 and L3 entity names/texts from Neo4j
2. Embed L2 (small, fits in VRAM)
3. Embed L3 using `SentenceTransformer all-MiniLM-L6-v2` — writes to `l3_embeddings.dat` **memory-mapped file** so it can resume if interrupted
4. Chunked GPU matmul: `(num_l2, emb_dim) @ (emb_dim, l3_chunk)` → finds Top-1 best L3 match per L2 entity
5. Filters by threshold 0.60, pushes `the_definition_of` edges to Neo4j in batches of 2000

**Designed to handle 3.5M L3 nodes** without OOM by chunking L3 in slices of 100k.

---

### `pubmed_importer.py` — Bulk PubMed Layer 2 Seeder (226 lines)

Standalone bulk importer. **No OpenAI used** — relies on PubMed's own MeSH annotations.

**Pipeline per keyword:**
1. `esearch` → get PMIDs
2. `efetch` → full XML with `rettype=xml`
3. Parse `<ArticleTitle>`, `<AbstractText>`, `<MeshHeading>` elements
4. Insert article nodes + MeSH term nodes + `HAS_MESH` relationships into Neo4j as L2 (`PubMed_Entity`)

Contains `PUBMED_KEYWORDS` — a list of ~100 broad medical keywords spanning diseases, drugs, procedures, and public health topics.

---

### `umls_importer.py` — Local UMLS Dump Importer (185 lines)

Parses the official UMLS Release files:
- `MRCONSO.RRF` — Concept names (filters to English only)
- `MRSTY.RRF` — Semantic type per CUI
- `MRREL.RRF` — Relationships between CUIs

**Streaming approach**: processes line-by-line with a batch buffer of 5,000 records to avoid RAM exhaustion on 3.5M+ concepts.

---

### `medical_terms.py` — Curated Seed List (56 lines)

A hand-curated list of ~170 medical terms (drugs, diseases, anatomy, procedures, drug classes) exported via `get_medical_terms()`. Used for bulk-seeding Layer 3 via the UMLS API.

---

### `patch_app.py` / `patch_med.py` — One-off Code Patches

Regex-based scripts that were used to add new features post-hoc (adding "Clear Relationships Only" button and method). These are historical artifacts — their changes are already applied to `app.py` and `med_graph_rag.py`.

---

### `lib/` — Frontend JS Libraries

Static JS libraries for PyVis graph rendering:
- `vis-9.1.2/` — Vis.js network visualization
- `tom-select/` — dropdown UI
- `bindings/` — PyVis HTML bindings

---

## 🔄 Complete Data Flow

```
User pastes text (or URL)
         │
         ▼
RecursiveCharacterTextSplitter (1200 chars, 150 overlap)
         │
  ┌──────┴──────────────────────────┐
  │ ThreadPoolExecutor (5 workers)  │ ← per chunk, parallel
  │                                  │
  │  _extract_entities (LLM)         │
  │  _extract_relationships (LLM)    │
  │  entity.embedding = embed(text)  │
  └──────────────────────────────────┘
         │
  NetworkX.DiGraph + Neo4j sync (L1 entities + rels)
         │
  _link_layers() ← vectorized NumPy cosine (L1→L2)
         │
  _tag_all_graphs() ← parallel LLM tagging
         │
  _build_tag_tree() ← agglomerative clustering on tag embeddings
         │
  Neo4j TagNode sync
         │
         ✅ Graph ready
         
User asks question
         │
  _top_down_retrieve() ← traverse tag_tree, find best leaf MetaMedGraph
         │
  Score top-8 entities by cosine sim to query
         │
  Neo4j k-hop traversal (k=2) → up to 50 triple neighbours
         │
  _generate_answer() ← LLM: entities + graph rels + neighbours
         │
  collect_ancestor_summaries() ← climb tag tree
         │
  _refine_answer() ← LLM per ancestor level (bottom-up)
         │
         ✅ Final answer returned
```

---

## ⚙️ Environment & Dependencies

| Category | Libraries |
|----------|-----------|
| **LLM** | `langchain-openai`, `openai` → `gpt-4o-mini` |
| **Embeddings** | `langchain-huggingface`, `sentence-transformers` → `all-MiniLM-L6-v2` (384-dim) |
| **Graph DB** | `neo4j==6.1.0`, `networkx` |
| **ML/Math** | `numpy`, `scipy`, `scikit-learn`, `torch` (optional GPU) |
| **UI** | `streamlit==1.56.0`, `pyvis` |
| **Web/APIs** | `requests`, `langchain-community` (WebBaseLoader) |
| **Text Split** | `langchain-text-splitters` |

**Required `.env` keys:**
```
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
UMLS_API_KEY=...          # optional, for live UMLS lookup
HUGGINGFACEHUB_API_TOKEN=...  # optional
```

---

## 🐛 Known Issues & Design Decisions

| Issue | Status |
|-------|--------|
| ThreadPoolExecutor on Windows can cause crashes during ingestion | Mitigated with `max_workers=5` |
| `neo4j_creds` attribute missing in `import_local_umls_dump` (references `self.neo4j_creds` but only `self.neo4j` is set in `__init__`) | **Active bug** |
| L3 RAM is zero but `repo_entities_l3` list still exists as a vestigial attribute | Legacy from earlier design |
| Tag tree top-down traversal falls back to `meta_graphs[0]` too eagerly | Can be improved |
| `patch_app.py` / `patch_med.py` are historical one-off scripts left in root | Can be removed |
| Cross-layer L1→L3 direct linking is not implemented (only L1→L2, L2→L3 via GPU) | By design (paper architecture) |

---

## 📊 Scale Capabilities

- **Layer 1**: Typically 5–30 chunks per document ingestion
- **Layer 2**: 14,000+ PubMed articles + MeSH terms  
- **Layer 3**: Up to **3,480,704 UMLS concepts** in Neo4j (Zero-RAM)
- **Embedding Batch Size**: 256 for L2 backfill
- **GPU Linker L3 Chunk Size**: 100,000 vectors per matmul pass
- **Neo4j Sync Batch Size**: 1,000 entities / relationships per transaction

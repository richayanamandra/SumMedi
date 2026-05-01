"""
MedGraphRAG — Triple Graph Construction + U-Retrieval
Based on: "Medical Graph RAG: Towards Safe Medical Large Language Model
           via Graph Retrieval-Augmented Generation" (Wu et al., 2024)
"""

from __future__ import annotations
from typing import Optional
import networkx as nx
import numpy as np

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_models import Entity, Relationship, MetaMedGraph, MEDICAL_TAGS, BUILTIN_VOCAB
from api_clients import UMLSClient, Neo4jClient
from llm_helpers import (
    EmbeddingStore, 
    _call_llm_json, 
    _extract_entities, 
    _extract_relationships, 
    _tag_graph, 
    _generate_answer, 
    _refine_answer
)

class MedGraphRAG:
    """
    Implements the three-layer graph + U-Retrieval from the MedGraphRAG paper.

    Graph layers
    ------------
    Layer 1 : Meta-MedGraphs  (one per semantic chunk of user documents)
    Layer 2 : Repository graph – Med Papers/Books entities
    Layer 3 : Repository graph – Medical Vocabulary / UMLS-style controlled vocab

    Cross-layer links are built by cosine-similarity thresholding.
    """

    SIMILARITY_THRESHOLD = 0.45   # δr  — cross-layer linking threshold
    TAG_MERGE_THRESHOLD  = 0.60   # δt  — hierarchical tag clustering threshold
    TOP_N_ENTITIES       = 8      # Nu  — entities retrieved per query
    TOP_K_NEIGHBOURS     = 2      # ku  — triple-neighbour hops
    MAX_TAG_LAYERS       = 6      # max U-Retrieval layers

    def __init__(
        self, 
        llm: BaseChatModel, 
        embedder: Embeddings, 
        umls_api_key: Optional[str] = None,
        neo4j_creds: Optional[dict] = None,
    ):
        self.llm = llm
        self.emb = EmbeddingStore(embedder)
        self.umls = UMLSClient(api_key=umls_api_key)
        
        # Neo4j setup
        self.neo4j = None
        if neo4j_creds:
            self.neo4j = Neo4jClient(
                uri=neo4j_creds.get("uri"),
                user=neo4j_creds.get("user"),
                password=neo4j_creds.get("password")
            )

        # Layer 1 – user RAG graphs
        self.meta_graphs: list[MetaMedGraph] = []

        # Hierarchical tag tree  [{graph_id: str, tags: dict, children: list}]
        self.tag_tree: list[dict] = []

        # NetworkX graph for Layer 1 only (keeps UI snappy for specific documents)
        self.nx_graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Layer 3 – Vocabulary (UMLS-style) constant layer initialization
    # ------------------------------------------------------------------

    def _build_vocab_layer(self) -> list[Entity]:
        entities = []
        for v in BUILTIN_VOCAB:
            e = Entity(
                name=v["name"],
                entity_type=v["type"],
                context=v["definition"],
                definition=v["definition"],
                layer=3,
            )
            entities.append(e)
        return entities

    def bulk_seed_vocabulary(self, terms: list[str], progress_callback=None):
        """
        Seed Layer 3 with a specific list of medical terms.
        Falls back to BUILTIN_VOCAB if UMLS fails.
        """
        if not self.neo4j:
            print("⚠️ Neo4j not connected. Seeding vocabulary in-memory only (limited).")
            return

        total = len(terms)
        for i, term in enumerate(terms):
            # Duplicate check – Database Native
            if self.neo4j.get_entity_by_name(term):
                if progress_callback:
                    progress_callback(i + 1, total, f"⏩ Skipping (exists in DB): {term}")
                continue

            details = None
            if self.umls.api_key:
                if progress_callback:
                    progress_callback(i + 1, total, f"🔍 UMLS Seeding: {term}")
                details = self.umls.get_term_details(term)
            
            if details:
                u_ent = Entity(
                    name=details["name"],
                    entity_type=details["type"],
                    context=details["definition"] or f"UMLS Concept: {details['name']}",
                    definition=details["definition"],
                    layer=3,
                )
                self.neo4j.sync_entities([u_ent])
            else:
                # Fallback to BUILTIN_VOCAB
                for vocab in BUILTIN_VOCAB:
                    if vocab["name"].lower() == term.lower():
                        if progress_callback:
                            progress_callback(i + 1, total, f"🔍 BUILTIN fallback: {term}")
                        u_ent = Entity(
                            name=vocab["name"],
                            entity_type=vocab["type"],
                            context=vocab["definition"],
                            definition=vocab["definition"],
                            layer=3,
                        )
                        self.neo4j.sync_entities([u_ent])
                        break

    # ------------------------------------------------------------------
    # Step 1 – Semantic document chunking
    # ------------------------------------------------------------------

    def _semantic_chunks(self, text: str, chunk_size: int = 1200) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " "],
        )
        return splitter.split_text(text)

    # ------------------------------------------------------------------
    # Step 2+4 – Entity extraction + relationship linking → MetaMedGraph
    # ------------------------------------------------------------------

    def _build_meta_graph(self, chunk_text: str, graph_id: str) -> MetaMedGraph:
        g = MetaMedGraph(graph_id=graph_id, source_text=chunk_text)
        g.entities = _extract_entities(self.llm, chunk_text)
        g.relationships = _extract_relationships(self.llm, chunk_text, g.entities)

        # Embed entities
        for e in g.entities:
            e.embedding = self.emb.embed(e.content_text)

        # Add to NetworkX
        for e in g.entities:
            self.nx_graph.add_node(e.name, entity=e)
        for r in g.relationships:
            self.nx_graph.add_edge(r.source, r.target, relation=r.relation)

        return g

    # ------------------------------------------------------------------
    # Step 3 – Triple Linking  (L1 → L2 → L3)
    # ------------------------------------------------------------------

    def _link_layers(self):
        """
        Hyper-scalable linking: For every Layer-1 entity, query Neo4j 
        to find Layer-2 and Layer-3 documents/concepts.
        """
        if not self.neo4j: return

        # L1 → L2 & L1 → L3
        for mg in self.meta_graphs:
            for e1 in mg.entities:
                # Find similar in L2 (papers) using Vector Search
                l2_sims = self.neo4j.find_similar_entities(e1.name, layer=2, limit=2, vector=e1.embedding.tolist())
                for e2 in l2_sims:
                    sim = self.emb.similarity(e1.embedding, self.emb.embed(e2.content_text))
                    # Sync to DB
                    self.neo4j.add_cross_layer_edge(e1.name, e2.name, "the_reference_of", sim)
                    # Sync to local NetworkX for UI
                    self.nx_graph.add_node(e2.name, entity=e2)
                    self.nx_graph.add_edge(e1.name, e2.name, relation="the_reference_of", similarity=sim)
                
                # Find similar in L3 (vocab) using Vector Search
                l3_sims = self.neo4j.find_similar_entities(e1.name, layer=3, limit=3, vector=e1.embedding.tolist())
                for e3 in l3_sims:
                    sim = self.emb.similarity(e1.embedding, self.emb.embed(e3.content_text))
                    # Sync to DB
                    self.neo4j.add_cross_layer_edge(e1.name, e3.name, "the_definition_of", sim)
                    # Sync to local NetworkX for UI
                    self.nx_graph.add_node(e3.name, entity=e3)
                    self.nx_graph.add_edge(e1.name, e3.name, relation="the_definition_of", similarity=sim)

    # ------------------------------------------------------------------
    # Step 5 – Tag the graphs  (hierarchical clustering)
    # ------------------------------------------------------------------

    def _tag_all_graphs(self):
        for mg in self.meta_graphs:
            mg.tag_summary = _tag_graph(self.llm, mg)

    def _build_tag_tree(self):
        """
        Agglomerative hierarchical clustering over tag embeddings.
        Produces a tree list used for top-down retrieval.
        """
        # Start: each graph is its own leaf node
        nodes = [
            {"ids": [mg.graph_id], "tags": mg.tag_summary, "children": []}
            for mg in self.meta_graphs
        ]

        for _layer in range(self.MAX_TAG_LAYERS):
            if len(nodes) <= 1:
                break

            # Compute pairwise tag similarities
            best_sim, best_i, best_j = -1.0, 0, 1
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    sim = self._tag_similarity(nodes[i]["tags"], nodes[j]["tags"])
                    if sim > best_sim:
                        best_sim, best_i, best_j = sim, i, j

            if best_sim < self.TAG_MERGE_THRESHOLD:
                break  # nothing more to merge

            # Merge nodes[best_i] and nodes[best_j]
            merged_tags = self._merge_tags(nodes[best_i]["tags"], nodes[best_j]["tags"])
            merged = {
                "ids": nodes[best_i]["ids"] + nodes[best_j]["ids"],
                "tags": merged_tags,
                "children": [nodes[best_i], nodes[best_j]],
            }
            new_nodes = [n for k, n in enumerate(nodes) if k not in (best_i, best_j)]
            new_nodes.append(merged)
            nodes = new_nodes

        self.tag_tree = nodes  # list of root-level nodes

    def _tag_similarity(self, tags_a: dict[str, str], tags_b: dict[str, str]) -> float:
        if not tags_a or not tags_b:
            return 0.0
        sims = []
        for ta_val in tags_a.values():
            for tb_val in tags_b.values():
                sims.append(self.emb.similarity(ta_val, tb_val))
        return float(np.mean(sims)) if sims else 0.0

    def _merge_tags(self, tags_a: dict[str, str], tags_b: dict[str, str]) -> dict[str, str]:
        merged = {}
        all_keys = set(tags_a) | set(tags_b)
        for k in all_keys:
            parts = []
            if k in tags_a:
                parts.append(tags_a[k])
            if k in tags_b:
                parts.append(tags_b[k])
            merged[k] = "; ".join(parts)
        return merged

    # ------------------------------------------------------------------
    # Step 6 – U-Retrieval
    # ------------------------------------------------------------------

    def _top_down_retrieve(self, query: str) -> MetaMedGraph | None:
        """
        Generate query tags, then traverse the tag tree top-down to find
        the most relevant Meta-MedGraph.
        """
        q_tags = _tag_graph(self.llm, MetaMedGraph(
            graph_id="query",
            entities=[Entity(name="query", entity_type="Other", context=query)],
        ))

        if not self.tag_tree:
            return self.meta_graphs[0] if self.meta_graphs else None

        # Traverse from root nodes
        current_nodes = self.tag_tree
        target_graph_id = None

        for _ in range(self.MAX_TAG_LAYERS):
            best_sim, best_node = -1.0, None
            for node in current_nodes:
                sim = self._tag_similarity(q_tags, node["tags"])
                if sim > best_sim:
                    best_sim, best_node = sim, node

            if best_node is None:
                break

            if not best_node["children"]:
                # Leaf: pick the graph
                target_graph_id = best_node["ids"][0]
                break
            else:
                current_nodes = best_node["children"]

        if target_graph_id is None and self.meta_graphs:
            target_graph_id = self.meta_graphs[0].graph_id

        for mg in self.meta_graphs:
            if mg.graph_id == target_graph_id:
                return mg
        return self.meta_graphs[0] if self.meta_graphs else None

    def _get_triple_neighbours(
        self, entity_name: str, k: int
    ) -> list[Entity]:
        """
        Neo4j-backed neighborhood traversal.
        """
        if self.neo4j:
            return self.neo4j.get_neighbors(entity_name, hops=k)
        return []

    def query(self, question: str) -> dict:
        """
        Full U-Retrieval pipeline.
        Returns a dict with keys: answer, target_graph, top_entities,
        triple_neighbours, refinement_log.
        """
        if not self.meta_graphs:
            return {"answer": "No documents loaded.", "target_graph": None,
                    "top_entities": [], "triple_neighbours": [], "refinement_log": []}

        # Top-down retrieval
        target_mg = self._top_down_retrieve(question)

        # Embed query and retrieve top-N entities from target graph
        q_emb = self.emb.embed(question)
        scored = []
        for e in target_mg.entities:
            if e.embedding is not None:
                sim = self.emb.similarity(q_emb, e.embedding)
                scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_entities = [e for _, e in scored[: self.TOP_N_ENTITIES]]

        # Gather triple neighbours (cross-layer)
        triple_neighbours: list[Entity] = []
        seen = {e.name for e in top_entities}
        for e in top_entities:
            for nb in self._get_triple_neighbours(e.name, self.TOP_K_NEIGHBOURS):
                if nb.name not in seen:
                    seen.add(nb.name)
                    triple_neighbours.append(nb)

        # Initial bottom-level answer
        answer = _generate_answer(
            self.llm, question, target_mg, top_entities, triple_neighbours
        )

        refinement_log = [{"level": 0, "answer": answer}]

        # Bottom-up refinement through higher tag layers
        def collect_ancestor_summaries(nodes, target_id, depth=0) -> list[tuple[int, dict]]:
            results = []
            for node in nodes:
                if target_id in node["ids"]:
                    results.append((depth, node["tags"]))
                    if node["children"]:
                        results += collect_ancestor_summaries(
                            node["children"], target_id, depth + 1
                        )
            return results

        ancestors = collect_ancestor_summaries(self.tag_tree, target_mg.graph_id)
        # Sort by depth ascending (higher-level first for bottom-up)
        ancestors.sort(key=lambda x: x[0])

        for level, summary in ancestors[1:]:  # skip level 0 (the graph itself)
            answer = _refine_answer(self.llm, question, answer, summary)
            refinement_log.append({"level": level, "answer": answer})

        return {
            "answer": answer,
            "target_graph": target_mg,
            "top_entities": top_entities,
            "triple_neighbours": triple_neighbours,
            "refinement_log": refinement_log,
        }

    # ------------------------------------------------------------------
    # Public API: load documents
    # ------------------------------------------------------------------

    def load_documents(
        self,
        user_text: str,
        paper_texts: list[str] | None = None,
        progress_callback=None,
    ) -> dict:
        """
        Build the full triple-graph from user documents (Layer 1) and
        optional medical paper texts (Layer 2).

        Returns a summary of what was built.
        """
        # --- Selective clearing ---
        # We only clear L1 (the patient session) to avoid wiping the seeded L2/L3 repository.
        self.meta_graphs.clear()
        
        # Remove only L1 nodes from current NetworkX session
        nodes_to_remove = [
            n for n, d in self.nx_graph.nodes(data=True)
            if d.get("entity") and d.get("entity").layer == 1
        ]
        self.nx_graph.remove_nodes_from(nodes_to_remove)

        if self.neo4j:
            # Only delete L1 entities from Neo4j for this session
            with self.neo4j.driver.session() as session:
                session.run("MATCH (n:Entity) WHERE n.layer = 1 DETACH DELETE n")

        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        # --- Layer 1: User documents ---
        _progress("⚙️  Chunking user documents…")
        chunks = self._semantic_chunks(user_text)

        for i, chunk in enumerate(chunks):
            _progress(f"⚙️  Building Layer-1 graph for chunk {i+1}/{len(chunks)}…")
            gid = f"chunk_{i}"
            mg = self._build_meta_graph(chunk, gid)
            self.meta_graphs.append(mg)
            
            # Sync L1 to Neo4j
            if self.neo4j:
                self.neo4j.sync_entities(mg.entities)
                self.neo4j.sync_relationships(mg.relationships)
            


        # --- Layer 2: Medical paper entities (if provided) ---
        if paper_texts:
            for j, paper in enumerate(paper_texts):
                _progress(f"⚙️  Extracting Layer-2 entities from paper {j+1}/{len(paper_texts)}…")
                paper_entities = _extract_entities(self.llm, paper[:3000])
                for e in paper_entities:
                    e.layer = 2
                    e.embedding = self.emb.embed(e.content_text)
                
                # Sync L2 to Neo4j – No local list extend
                if self.neo4j:
                    self.neo4j.sync_entities(paper_entities)
                


        # --- Triple Linking ---
        _progress("🔗  Triple linking across graph layers…")
        # Links L1 entries to the existing L2 (PubMed) and L3 (UMLS) repository in Neo4j
        self._link_layers()

        # --- Tag graphs ---
        _progress("🏷️  Tagging graphs…")
        self._tag_all_graphs()

        # --- Build tag hierarchy ---
        _progress("🌲  Building hierarchical tag tree…")
        self._build_tag_tree()

        _progress("✅  Triple graph construction complete.")

        total_l1_entities = sum(len(mg.entities) for mg in self.meta_graphs)
        total_l1_rels = sum(len(mg.relationships) for mg in self.meta_graphs)
        cross_layer_edges = sum(
            1 for _, _, d in self.nx_graph.edges(data=True)
            if d.get("relation") in ("the_reference_of", "the_definition_of")
        )

        # Pull counts from Neo4j if available, otherwise fallback to 0
        l2_count = self.neo4j.get_layer_count(2) if self.neo4j else 0
        l3_count = self.neo4j.get_layer_count(3) if self.neo4j else 0

        return {
            "chunks": len(chunks),
            "meta_graphs": len(self.meta_graphs),
            "l1_entities": total_l1_entities,
            "l1_relationships": total_l1_rels,
            "l2_entities": l2_count,
            "l3_entities": l3_count,
            "cross_layer_edges": cross_layer_edges,
            "tag_tree_roots": len(self.tag_tree),
            "total_graph_nodes": self.nx_graph.number_of_nodes(),
            "total_graph_edges": self.nx_graph.number_of_edges(),
        }

    def get_graph_stats(self) -> dict:
        if self.neo4j:
            return {
                "meta_graphs": len(self.meta_graphs),
                "l1_entities": self.neo4j.get_layer_count(1),
                "l2_entities": self.neo4j.get_layer_count(2),
                "l3_entities": self.neo4j.get_layer_count(3),
                "total_nodes": self.neo4j.get_layer_count(1) + self.neo4j.get_layer_count(2) + self.neo4j.get_layer_count(3),
                "total_edges": 0 # TODO: count relationships in DB
            }
        return {
            "meta_graphs": len(self.meta_graphs),
            "l1_entities": sum(len(mg.entities) for mg in self.meta_graphs),
            "l2_entities": 0,
            "l3_entities": 0,
            "total_nodes": self.nx_graph.number_of_nodes(),
            "total_edges": self.nx_graph.number_of_edges(),
        }

    def clear_all(self):
        """
        Clears all graph data globally from memory and Neo4j.
        """
        self.meta_graphs.clear()
        self.tag_tree.clear()
        self.nx_graph.clear()
        if self.neo4j:
            self.neo4j.clear_all_db()

    def clear_all_relationships(self):
        """
        Clears all relationships (edges) from memory and Neo4j, but retains nodes.
        """
        self.nx_graph.clear_edges()
        if self.neo4j:
            self.neo4j.clear_all_relationships()

    def simulate_massive_vocab(self, num_nodes=500000, batch_size=10000, progress_callback=None):
        """
        Injects a massive set of dummy nodes into Layer 3 to simulate high capacity.
        Processed in batches to avoid OOM in python list and Neo4j.
        Computes Hugging Face embeddings in batches if configured.
        """
        self.clear_all()
        
        total_batches = (num_nodes // batch_size) + (1 if num_nodes % batch_size else 0)
        
        for batch_i in range(total_batches):
            if progress_callback:
                progress_callback(batch_i + 1, total_batches, f"Injecting batch {batch_i + 1}/{total_batches} ({num_nodes} nodes total) into L3")
            
            start_i = len(self.repo_entities_l3)
            end_i = min(start_i + batch_size, num_nodes)
            
            if start_i >= num_nodes:
                break
                
            batch_entities = []
            for i in range(start_i, end_i):
                e = Entity(
                    name=f"SimNode_{i}",
                    entity_type="Simulated",
                    context=f"Simulated entity {i} for stress test",
                    definition=f"Definition of simulated entity {i}",
                    layer=3
                )
                batch_entities.append(e)
            
            # Embed batch
            texts = [e.content_text for e in batch_entities]
            try:
                embeddings = self.emb.embed_batch(texts)
                for e, vec in zip(batch_entities, embeddings):
                    e.embedding = vec
            except Exception as e:
                print(f"Embedding batch failed: {e}")
            
            if self.neo4j:
                self.neo4j.sync_entities(batch_entities)
                # Ensure we add mock relationships sequentially so the graph structure exists.
                # E.g., SimNode_i -> SimNode_i+1
                rels = []
                for i in range(len(batch_entities) - 1):
                    rels.append(Relationship(
                        source=batch_entities[i].name,
                        target=batch_entities[i+1].name,
                        relation="SIMULATED_LINK"
                    ))
                self.neo4j.sync_relationships(rels)

    def import_local_umls_relationships_dump(self, mrconso_path: str, mrrel_path: str, progress_callback=None):
        from umls_importer import load_umls_relationships_to_neo4j
        if not self.neo4j_creds:
            raise ValueError("Neo4j credentials are required to import UMLS dump.")
            
        uri = self.neo4j_creds.get("uri")
        user = self.neo4j_creds.get("user")
        pwd = self.neo4j_creds.get("password")
        
        return load_umls_relationships_to_neo4j(mrconso_path, mrrel_path, uri, user, pwd, progress_callback)

    def import_local_umls_dump(self, mrconso_path: str, mrsty_path: str, progress_callback=None):
        from umls_importer import load_umls_to_neo4j
        if not self.neo4j_creds:
            raise ValueError("Neo4j credentials are required to import UMLS dump.")
            
        uri = self.neo4j_creds.get("uri")
        user = self.neo4j_creds.get("user")
        pwd = self.neo4j_creds.get("password")
        
        return load_umls_to_neo4j(mrconso_path, mrsty_path, uri, user, pwd, progress_callback)

    def load_reference_papers(self, paper_texts: list[str], progress_callback=None, max_workers: int = 1):
        """
        Additively load medical paper entities into Layer 2 with batch optimization and optional parallelism.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)

        total = len(paper_texts)
        if total == 0:
            return

        def process_paper(paper):
            try:
                # 1. Extract entities (truncate to avoid token limits)
                paper_entities = _extract_entities(self.llm, paper[:4000])
                for e in paper_entities:
                    e.layer = 2
                
                # 2. Batch embedding for this paper's entities
                texts = [e.context for e in paper_entities]
                embeddings = self.emb.embed_batch(texts)
                for e, vec in zip(paper_entities, embeddings):
                    e.embedding = vec
                
                # 3. Extract internal relationships for Layer 2
                paper_relationships = _extract_relationships(self.llm, paper[:4000], paper_entities)
                
                return paper_entities, paper_relationships
            except Exception as e:
                return e

        all_new_entities = []
        all_new_rels = []
        _progress(f"🚀 Starting parallel ingestion with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {executor.submit(process_paper, paper): i for i, paper in enumerate(paper_texts)}
            
            for future in as_completed(future_to_paper):
                i = future_to_paper[future]
                result = future.result()
                
                if isinstance(result, tuple) and len(result) == 2:
                    entities, relationships = result
                    all_new_entities.extend(entities)
                    all_new_rels.extend(relationships)
                    _progress(f"✅ Processed paper {i+1}/{total} ({len(entities)} ents, {len(relationships)} rels)")
                else:
                    _progress(f"⚠️  Error on paper {i+1}: {result}")

                # Periodic sync to Neo4j to keep memory low
                if len(all_new_entities) >= 100:
                    if self.neo4j:
                        self.neo4j.sync_entities(all_new_entities)
                        self.neo4j.sync_relationships(all_new_rels)
                    all_new_entities = []
                    all_new_rels = []
        
        # Final sync
        if self.neo4j:
            if all_new_entities:
                self.neo4j.sync_entities(all_new_entities)
            if all_new_rels:
                self.neo4j.sync_relationships(all_new_rels)
        
        _progress(f"🎉 Finished importing {total} papers into Layer 2.")

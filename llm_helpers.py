import json
import re
import textwrap
import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from scipy.spatial.distance import cosine

from data_models import Entity, Relationship, MetaMedGraph, UMLS_SEMANTIC_TYPES, MEDICAL_TAGS

from typing import Any

import threading

class EmbeddingStore:
    def __init__(self, embedder: Any):
        self._emb = embedder
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def embed(self, text: str) -> np.ndarray:
        with self._lock:
            if text in self._cache:
                return self._cache[text]
        
        vec = self._emb.embed_query(text)
        arr = np.array(vec, dtype=np.float32)
        
        with self._lock:
            self._cache[text] = arr
        return arr

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        with self._lock:
            uncached = [t for t in set(texts) if t and t not in self._cache]
        
        if uncached:
            if hasattr(self._emb, 'embed_documents'):
                vecs = self._emb.embed_documents(uncached)
                with self._lock:
                    for t, v in zip(uncached, vecs):
                        self._cache[t] = np.array(v, dtype=np.float32)
            else:
                for t in uncached:
                    self.embed(t)
        
        with self._lock:
            return [self._cache[t] if t else None for t in texts]


    def similarity(self, a: str | np.ndarray, b: str | np.ndarray) -> float:
        va = a if isinstance(a, np.ndarray) else self.embed(a)
        vb = b if isinstance(b, np.ndarray) else self.embed(b)
        if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
            return 0.0
        return float(1.0 - cosine(va, vb))


def _call_llm_json(llm: BaseChatModel, prompt: str) -> dict | list:
    """Call LLM and parse JSON from the response with robustness for small models."""
    try:
        resp = llm.invoke(prompt)
        raw = resp.content.strip()
    except Exception as e:
        print(f"LLM Call Error: {e}")
        return []
    
    # Pre-clean: remove markdown fences
    raw = re.sub(r"```(json)?", "", raw, flags=re.IGNORECASE)
    raw = raw.strip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback 1: Extract anything between the first [ and last ] or first { and last }
        # This handles models that add conversational text around the JSON
        list_match = re.search(r"(\[.*\])", raw, re.DOTALL)
        dict_match = re.search(r"(\{.*\})", raw, re.DOTALL)
        
        target = None
        if list_match and dict_match:
             target = list_match.group(1) if list_match.start() < dict_match.start() else dict_match.group(1)
        elif list_match:
             target = list_match.group(1)
        elif dict_match:
             target = dict_match.group(1)
            
        if target:
            try:
                # Cleanup: handle common small model mishaps
                clean = re.sub(r",\s*([\]\}])", r"\1", target) # trailing commas
                return json.loads(clean)
            except:
                pass
        
        return []


def _extract_entities(llm: BaseChatModel, chunk_text: str) -> list[Entity]:
    semantic_types_str = ", ".join(UMLS_SEMANTIC_TYPES)
    prompt = textwrap.dedent(f"""
        You are a biomedical NLP expert. Extract all medically relevant entities from the text below.

        FORMAT RULES:
        - Return ONLY a JSON array.
        - "name": Use the specific medical/chemical name (e.g., "Metformin").
        - "type": Choose ONLY from this list: [{semantic_types_str}]
        - "context": A 1-2 sentence description from the text.

        EXAMPLES:
        Input: "Patient is taking 500mg of Aspirin for heart pain."
        Output: [
            {{"name": "Aspirin", "type": "Pharmacologic Substance", "context": "Patient takes 500mg for heart pain"}},
            {{"name": "Heart pain", "type": "Finding", "context": "Reason for taking aspirin"}}
        ]

        Input: "She has been diagnosed with Malaria."
        Output: [
            {{"name": "Malaria", "type": "Disease or Syndrome", "context": "The patient's primary diagnosis"}}
        ]

        TEXT:
        {chunk_text}
    """).strip()
    result = _call_llm_json(llm, prompt)
    entities = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "name" in item:
                entities.append(Entity(
                    name=item.get("name", "Unknown"),
                    entity_type=item.get("type", "Other"),
                    context=item.get("context", ""),
                    layer=1,
                ))
    return entities


def _extract_relationships(
    llm: BaseChatModel, chunk_text: str, entities: list[Entity]
) -> list[Relationship]:
    entity_names = [e.name for e in entities]
    if len(entity_names) < 2:
        return []
    prompt = textwrap.dedent(f"""
        You are a biomedical knowledge graph expert.
        Given the entities: {entity_names}
        And the source text below, identify meaningful relationships BETWEEN those entities.

        Return ONLY a JSON array where each element has:
          - "source": name of source entity (must be from the list above)
          - "relation": a short relation phrase (e.g. "treats", "causes", "is_symptom_of")
          - "target": name of target entity (must be from the list above)

        Only include relationships explicitly or strongly implied by the text.
        Return ONLY the JSON array, no markdown, no explanation.

        TEXT:
        {chunk_text}
    """).strip()
    result = _call_llm_json(llm, prompt)
    rels = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "source" in item and "target" in item:
                rels.append(Relationship(
                    source=item.get("source", ""),
                    relation=item.get("relation", "related_to"),
                    target=item.get("target", ""),
                ))
    return rels


def _tag_graph(llm: BaseChatModel, graph: MetaMedGraph) -> dict[str, str]:
    entity_texts = "\n".join(
        f"- {e.name} ({e.entity_type}): {e.context}" for e in graph.entities
    )
    tags_str = ", ".join(MEDICAL_TAGS)
    prompt = textwrap.dedent(f"""
        You are a medical text summarizer. Summarize the following medical entities using
        these structured tag categories: {tags_str}

        For each relevant tag, provide a short phrase describing what is present.
        Return ONLY a JSON object where keys are tag names and values are short descriptions.
        Omit tags that are not relevant. No markdown, no explanation.

        ENTITIES:
        {entity_texts}
    """).strip()
    result = _call_llm_json(llm, prompt)
    if isinstance(result, dict):
        return {k: str(v) for k, v in result.items()}
    return {}


def _generate_answer(
    llm: BaseChatModel,
    question: str,
    graph: MetaMedGraph,
    top_entities: list[Entity],
    top_k_neighbors: list[Entity],
) -> str:
    graph_text = ""
    all_ents = {e.name: e for e in top_entities + top_k_neighbors}
    for rel in graph.relationships:
        if rel.source in all_ents or rel.target in all_ents:
            src_e = all_ents.get(rel.source)
            tgt_e = all_ents.get(rel.target)
            src_ctx = f" [{src_e.context[:80]}]" if src_e else ""
            tgt_ctx = f" [{tgt_e.context[:80]}]" if tgt_e else ""
            # Include definitions and sources from layers 2 & 3
            src_def = ""
            tgt_def = ""
            if src_e and src_e.layer == 3:
                src_def = f" (Definition: {src_e.definition})"
            if tgt_e and tgt_e.layer == 3:
                tgt_def = f" (Definition: {tgt_e.definition})"
            graph_text += (
                f"{rel.source}{src_ctx}{src_def} "
                f"--[{rel.relation}]--> "
                f"{rel.target}{tgt_ctx}{tgt_def}\n"
            )

    entity_detail = "\n".join(
        f"• {e.name} ({e.entity_type}, Layer {e.layer}): {e.context}"
        + (f"\n  Source/Definition: {e.definition}" if e.definition else "")
        for e in (top_entities + top_k_neighbors)
    )

    prompt = textwrap.dedent(f"""
        You are a medical expert assistant generating evidence-based responses.

        QUESTION: {question}

        RELEVANT ENTITIES (with source and definition references):
        {entity_detail}

        GRAPH RELATIONSHIPS:
        {graph_text if graph_text else "(no direct relationships found)"}

        Using the entities and graph above, answer the question in detail.
        Cite specific entities by name. If definitions are provided use them to
        clarify terminology. Be precise and evidence-based.
    """).strip()
    resp = llm.invoke(prompt)
    return resp.content.strip()


def _refine_answer(
    llm: BaseChatModel, question: str, prev_response: str, summary: dict[str, str]
) -> str:
    summary_text = "\n".join(f"  {k}: {v}" for k, v in summary.items())
    prompt = textwrap.dedent(f"""
        You are a medical expert assistant. Refine the response below using the
        higher-level summary context provided.

        QUESTION: {question}

        PREVIOUS RESPONSE:
        {prev_response}

        ADDITIONAL CONTEXT (higher-level summary):
        {summary_text}

        Adjust and improve the response, ensuring completeness and accuracy.
        Preserve all cited evidence from the previous response and add any new
        relevant information from the additional context.
    """).strip()
    resp = llm.invoke(prompt)
    return resp.content.strip()

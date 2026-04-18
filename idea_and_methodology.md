# Idea & Methodology — SumMedi

This document provides a detailed breakdown of the architecture, implementation strategy, and experimental design of the project.

---

## 🧠 Core Idea

Large Language Models (LLMs) struggle in medical domains due to:

- Limited context window  
- Hallucinations and lack of verifiability  
- Absence of structured domain knowledge  

Traditional RAG improves factual grounding but fails in:

- Multi-hop reasoning  
- Cross-document synthesis  

To address this, we implement and extend MedGraphRAG, a graph-based RAG pipeline that:

- Builds structured knowledge graphs from medical data  
- Links entities to credible sources + controlled vocabularies  
- Performs hierarchical retrieval + reasoning  

---

## 🏗️ Architecture

We follow the same 3-layer Graph RAG architecture, adapted to our dataset and experimentation setup:

---

### 1. Data Layer

- **Primary corpus:** PubMed Central (PMC)  
- Replaces the original 4.8M biomedical corpus  

**Provides:**
- Peer-reviewed medical literature  
- High-quality structured abstracts  

**Additional:**
- UMLS-style semantic medical vocabularies  

---

### 2. Graph Construction Layer

#### a. Semantic Chunking

- Hybrid approach:
  - Paragraph-based segmentation  
  - LLM-driven topic consistency  
- Sliding window ensures semantic coherence  

---

#### b. Entity Extraction

Each chunk → structured entities:

```python
Entity = {
  "name": ...,
  "type": "UMLS semantic types",
  "context": ...
}

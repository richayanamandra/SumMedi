# 🧠 **SumMedi — Medical Text Summarization using Multi-Layer GraphRAG**

## 📌 Abstract

Large Language Models (LLMs) have shown remarkable capabilities in natural language understanding but struggle significantly in **high-stakes domains like medicine**, where accuracy, explainability, and grounding in verified knowledge are critical. Traditional Retrieval-Augmented Generation (RAG) systems rely on flat vector retrieval, which lacks structured reasoning and often leads to hallucinations.

In this project, we present **SumMedi**, a **Multi-Layer Graph-based Retrieval-Augmented Generation (GraphRAG) system**, inspired by MedGraphRAG , designed to generate **evidence-based, explainable medical responses**.

Our system constructs a **hierarchical knowledge graph** combining:

* **User medical data (Layer 1)**
* **PubMed research corpus (Layer 2)**
* **UMLS ontology (Layer 3)**

We introduce a **practical implementation of Triple Graph Linking and U-Retrieval**, enabling:

* Multi-hop reasoning
* Cross-layer semantic grounding
* Context-aware summarization

The result is a system that significantly improves **trustworthiness, scalability, and interpretability** of medical AI.

---

# ⚠️ 1. Introduction

Recent advances in LLMs (e.g., GPT, Gemini, LLaMA) have enabled powerful language understanding. However, in medicine:

* Knowledge is **too large** for context windows
* Responses must be **factually correct and verifiable**
* Hallucinations can lead to **serious consequences**

As highlighted in MedGraphRAG :

> Medical AI requires reasoning over structured knowledge, not just text similarity.

---

## ❌ Limitations of Existing Approaches

### Traditional RAG

* Retrieves text chunks using vector similarity
* Cannot perform **multi-hop reasoning**
* Lacks **explainability**

### GraphRAG

* Builds knowledge graphs
* But:

  * Expensive graph construction
  * Weak grounding to **authoritative medical sources**

---

## ✅ Our Approach

We propose **SumMedi**, a system that:

* Uses **multi-layer knowledge graphs**
* Anchors reasoning to:

  * **UMLS (ground truth)**
  * **PubMed (evidence)**
* Implements **U-Retrieval** for efficient and context-aware querying

---

# 🏗️ 2. System Architecture

## 2.1 Overview

The system follows a **three-layer hierarchical graph architecture**:

### 🧾 Layer 1: User Data (Dynamic Graph)

* Input: medical notes / documents / URLs
* Output: **Meta-MedGraph**
* Contains:

  * Entities (disease, drug, symptom)
  * Relationships (treats, causes, etc.)

---

### 📚 Layer 2: Medical Literature (PubMed)

* ~10,000 research papers
* Provides:

  * Clinical validation
  * Evidence-based relationships

---

### 🧠 Layer 3: UMLS Ontology

* ~3.5 million concepts
* ~10M+ relationships
* Acts as:

  * **Ground-truth dictionary**
  * Controlled medical vocabulary

---

## 🔗 Cross-Layer Linking (Triple Graph Construction)

Inspired by MedGraphRAG :

Each entity forms a triple:

```
[User Entity] → [PubMed Evidence] → [UMLS Definition]
```

Linking is performed using:

* **Cosine similarity**
* Embeddings of:

  * entity name
  * type
  * context

This ensures:

* Evidence-backed reasoning
* Explainability

---

# ⚙️ 3. Methodology

## 3.1 Data Pipeline

### Step 1: Semantic Chunking

* Input text is split into:

  * ~1200 character chunks
  * Overlapping for context preservation

Inspired by semantic chunking approach .

---

### Step 2: Named Entity Recognition (NER)

* LLM extracts:

  * Diseases
  * Drugs
  * Symptoms

Output:

```
Entity = {name, type, context}
```

---

### Step 3: Relationship Extraction

* LLM identifies relations:

Example:

```
Metformin → treats → Diabetes
```

---

### Step 4: Graph Construction

* Each chunk → small graph
* Graphs merged into **Meta-MedGraph**

---

### Step 5: Tag Tree Generation

Instead of expensive graph clustering (used in GraphRAG), we use:

* **Tag-based summarization**
* Hierarchical structure

Tags include:

* Symptoms
* Medication
* Body Functions
* Patient History

This forms a **multi-layer Tag Tree**

---

### Step 6: Embedding & Storage

* Embedding model:

  * `all-MiniLM-L6-v2` (384-dim vectors)

* Stored in:

  * **Neo4j Graph DB**
  * With vector indexing

---

# 🗄️ 4. Database Design

## 4.1 UMLS Layer

* 3.5M nodes
* 10M+ edges
* Predefined ontology

---

## 4.2 PubMed Layer

* Papers stored as nodes
* Precomputed embeddings

---

## 4.3 Linking Strategy

Edges created using:

```
cosine_similarity(entity, pubmed) → threshold → edge
cosine_similarity(pubmed, umls) → threshold → edge
```

---

## 4.4 Key Features

* Fully persistent (Neo4j-based)
* No RAM bottlenecks
* Scalable to millions of nodes

---

# 🔍 5. Retrieval Mechanism — U-Retrieval

## 5.1 Motivation

Traditional retrieval:

* Flat
* Context-limited

U-Retrieval:

* **Hierarchical + iterative**

---

## ⬇️ Phase A: Top-Down Retrieval

* Query → embedding
* Compared with Tag Tree
* Navigate to most relevant leaf node

```
Root → Category → Subcategory → Chunk Graph
```

---

## 🔗 Phase B: Triple-Neighbour Extraction

Retrieve:

* Layer 1 → patient graph
* Layer 2 → PubMed evidence
* Layer 3 → UMLS definitions

This aligns with triple retrieval in MedGraphRAG .

---

## ⬆️ Phase C: Bottom-Up Refinement

* Traverse back up Tag Tree
* Merge summaries
* Refine answer iteratively

This ensures:

* Local precision
* Global context awareness

---

## 🧠 Final Context

The LLM receives:

* Query
* Triplets (L1 + L2 + L3)
* Tag summaries

→ Generates grounded response

---

# 🧪 6. Experimental Insights (From Paper)

According to MedGraphRAG:

* Improves accuracy by:

  * ~8–10% over RAG
* Outperforms:

  * GraphRAG
  * Fine-tuned medical LLMs

Human evaluation shows:

* Better:

  * Citation precision
  * Understandability
  * Reliability 

---

# 🚀 7. Key Contributions

### 🔬 Technical

* Multi-layer GraphRAG implementation
* Integration of:

  * UMLS
  * PubMed
* Tag-based hierarchical retrieval

---

### ⚙️ System Design

* Scalable Neo4j-based architecture
* Efficient embedding + indexing

---

### 🧠 Algorithmic

* U-Retrieval (Top-down + Bottom-up)
* Cross-layer semantic alignment

---

# 📊 8. Advantages Over Traditional RAG

| Feature        | Traditional RAG | SumMedi      |
| -------------- | --------------- | ------------ |
| Structure      | Flat            | Graph-based  |
| Reasoning      | Single-hop      | Multi-hop    |
| Explainability | Low             | High         |
| Hallucination  | High            | Reduced      |
| Context        | Limited         | Hierarchical |

---

# ⚠️ 9. Limitations

* Dependency on:

  * Entity extraction quality
* Graph construction latency
* Requires:

  * Preprocessing pipeline

---

# 🔮 10. Future Work

* Real-time graph updates
* Clinical validation
* Integration with hospital systems
* Improved relation extraction models

---

# 📚 11. References

* MedGraphRAG Paper 
* UMLS Ontology
* PubMed Central

---

# 🎯 Conclusion

SumMedi demonstrates that:

> **“Structured knowledge retrieval + LLMs = reliable medical AI.”**

By combining:

* Graph-based reasoning
* Multi-layer knowledge
* Hierarchical retrieval

we move closer to **safe, explainable, and scalable medical AI systems**.



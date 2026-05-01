# SUMMEDI Project: Advanced Medical Triple-layer Graph RAG Framework

This project implements an advanced, database-native Retrieval-Augmented Generation (RAG) architecture tailored specifically for the medical domain, ensuring safe, evidence-grounded clinical question answering by linking user queries to established medical literature and controlled vocabularies.

## The Problem:
General Large Language Models (LLMs) struggle with the highly specialized medical domain because they cannot fit the vast, continuously evolving medical knowledge base into their standard context windows, which frequently leads to them generating plausible but incorrect information (hallucinations). While standard Retrieval-Augmented Generation (RAG) helps provide external context, it struggles to synthesize holistic insights across complex, multi-layered medical documents. Even advanced graph-based RAG approaches are often overly complex, computationally expensive, and lack specific mechanisms to ensure that the generated medical responses are backed by credible, verifiable evidence and definitions.  

## The Solution:
The MedGraphRAG framework solves this by introducing a highly structured, domain-specific graph retrieval architecture designed for safety and traceability. As illustrated in the workflow image provided previously, it solves the problem through two primary innovations:  

- Triple-Layer Graph Construction: Instead of a flat vector database, the system builds a three-tiered knowledge graph. It ingests user documents (Layer 1) and explicitly links the extracted medical entities and relationships to peer-reviewed medical literature like PubMed (Layer 2) and controlled medical vocabularies like UMLS (Layer 3). This forces the LLM to ground its reasoning in verified sources and standardized definitions.
- Hierarchical U-Retrieval: To search this massive graph efficiently, the framework generates hierarchical "Tag Trees" that summarize the graph at various levels of abstraction. When a query is received, the system performs a "Top-Down Traversal" of the tag tree to quickly zero in on the most relevant information. It then gathers the specific cross-layer connections and performs a "Bottom-Up Refinement," synthesizing the localized evidence before passing it to the LLM to generate a final, evidence-grounded answer.

## 📄 Documentation
-(literature_review.md)
- [Idea and Methodology](idea_and_methodology.md)

## 🏗️ Architecture and Workflow

<img width="2816" height="1363" alt="architecture" src="https://github.com/user-attachments/assets/fa8d7bd4-f110-44ed-aaae-cc141feae65e" />


The SumMedi pipeline processes information in two primary phases:
1. **Data Ingestion & Graph Construction:** Unstructured medical text is ingested and semantically chunked. A Large Language Model extracts entities (diseases, drugs, symptoms) and their relationships, constructing localized graphs. These are merged, tagged hierarchically to form a Tag Tree, and stored persistently in a Neo4j graph database.
2. **Hierarchical U-Retrieval:** When a query is submitted, it is converted into an embedding vector. The system performs an efficient U-Retrieval [1]:
   - **Top-Down Traversal:** Compares the query vector against the Tag Tree to pinpoint the most relevant graph structures.
   - **Graph Retrieval:** Queries across Layer 1 (Meta-MedGraph nodes), Layer 2 (PubMed nodes), and Layer 3 (UMLS nodes).
   - **Bottom-Up Refinement:** Forms a localized context of triplets and neighboring nodes, then traverses back up the tree to refine the context before generating the final evidence-grounded LLM response.


## 📊 MedQA Benchmark Results
Our architecture was rigorously evaluated on the USMLE-style MedQA dataset, demonstrating substantial improvements in diagnostic reasoning and accuracy:
- **GPT-4** achieved **96%** accuracy, surpassing state-of-the-art benchmarks on MedQA with the GraphRAG pipeline.
- **Gemma4:31b-cloud** achieved **90%** accuracy when utilizing our medical graph pipeline, showing a significant improvement over its **83%** baseline accuracy without the pipeline.
- Implemented a highly scalable, triple-layer knowledge repository backed by Neo4j, seamlessly integrating over 3.5 million nodes from the Unified Medical Language System (UMLS)
- Efficient U-Retrieval ensures comprehensive global context awareness combined with precise, localized evidence extraction

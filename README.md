# SumMedi — Medical Text Summarization using Multi-Layer GraphRAG

This project is a research-driven implementation and extension of MedGraphRAG, a graph-based Retrieval-Augmented Generation (RAG) framework for safe, evidence-grounded medical QA systems.

## 🚀 What this project does

- Implements a **Graph-based RAG pipeline** for medical text understanding
- Replaces large proprietary datasets with **PubMed Central (PMC)**
- Evaluates performance across **multiple LLMs (GPT, Gemini, Gemma, LLaMA)**
- Focuses on **evidence-backed, explainable medical responses**

---

## 🎯 Key Contributions

- 🔬 **Multi-LLM Evaluation**  
  Systematic comparison across GPT-4, Gemini, Gemma, and LLaMA variants

- 📚 **Dataset Adaptation**  
  Uses **PubMed Central (PMC)** instead of large-scale proprietary corpora

- 🧠 **Graph-based Reasoning**  
  Implements **Multi-layer GraphRAG** with structured knowledge representation

- ⚖️ **Architecture vs Model Study**  
  Analyzes how retrieval architecture impacts performance across models

---

## 🏗️ High-Level Architecture
Documents → Graph Construction → Hierarchical Retrieval → LLM Response

- Structured knowledge graphs replace flat vector retrieval
- Enables **multi-hop reasoning** and **context-aware generation**

---

## ⚙️ Models Used

- GPT-4 and variants  
- Gemini models  
- Gemma  
- LLaMA models  

---

## 📊 What is achieved

- Improved **context grounding** over standard RAG  
- Better **multi-hop reasoning capabilities**  
- Reduced reliance on model scale via structured retrieval  
- Evidence-backed response generation  

---

## 📌 Why this matters

Medical AI systems require:
- High accuracy  
- Explainability  
- Trustworthy sources  

This project explores how **graph-based retrieval + LLMs** can move closer to reliable medical AI systems.

---

## 📂 Detailed Methodology

👉 See [`idea_and_methodology.md`](./idea_and_methodology.md) for full technical details.

---

## 📚 Reference

Medical Graph RAG: Towards Safe Medical LLM via Graph Retrieval-Augmented Generation

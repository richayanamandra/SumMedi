import os
import argparse
import time
import json
from Bio import Entrez
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from med_graph_rag import MedGraphRAG

# Config
load_dotenv()
Entrez.email = os.environ.get("NCBI_EMAIL", "your.email@example.com")
api_key = os.environ.get("NCBI_API_KEY", "")
if api_key and not api_key.startswith("your_"):
    Entrez.api_key = api_key
else:
    Entrez.api_key = None

CHECKPOINT_FILE = "pubmed_checkpoint.json"

def load_processed_ids():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return list(json.load(f))
    return []

def main():
    parser = argparse.ArgumentParser(description="Retroactively add internal relationships to existing Layer 2 papers.")
    parser.add_argument("--limit", type=int, default=600, help="Max papers to densify")
    parser.add_argument("--batch", type=int, default=20, help="Batch size for PubMed fetching")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for extraction")
    
    args = parser.parse_args()

    # Init RAG
    print(f"⏳ Initializing RAG Backend for Densification (with {args.workers} workers)...")
    llm = ChatOllama(model="gemma3:1b", temperature=0.1)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    neo4j_creds = {
        "uri": os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "password")
    }
    
    rag = MedGraphRAG(
        llm=llm,
        embedder=embedder,
        umls_api_key=os.environ.get("UMLS_API_KEY"),
        neo4j_creds=neo4j_creds
    )

    # 1. Get processed IDs
    all_processed = load_processed_ids()
    if not all_processed:
        print("❌ No processed papers found in checkpoint. Run seed_pubmed.py first.")
        return

    todo_ids = all_processed[:args.limit]
    print(f"🚀 Found {len(todo_ids)} papers to densify. Starting relationship extraction...")

    # 2. Main Loop (Batched)
    for i in range(0, len(todo_ids), args.batch):
        batch = todo_ids[i : i + args.batch]
        print(f"\n--- Densifying Batch {i//args.batch + 1}: {len(batch)} papers ---")
        
        try:
            # Re-fetch abstracts
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="abstract", retmode="text")
            data = handle.read()
            handle.close()
            
            abstracts = [p.strip() for p in data.split("\n\n") if len(p.strip()) > 100]
            
            # Process batch using the updated MedGraphRAG (Entities + Relationships)
            # MERGE in Neo4j will prevent duplicate entities but ADD the missing relationships
            rag.load_reference_papers(abstracts, progress_callback=lambda msg: print(f"  {msg}"), max_workers=args.workers)
            
            # Entrez rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"⚠️  Error in batch densification: {e}")
            continue

    print("\n🎉 Layer 2 Relationship Densification Complete!")

if __name__ == "__main__":
    main()

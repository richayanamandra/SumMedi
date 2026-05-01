import os
import time
import argparse
import json
from Bio import Entrez
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from med_graph_rag import MedGraphRAG

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

load_dotenv()
email = os.environ.get("NCBI_EMAIL", "your.email@example.com")
Entrez.email = email

api_key = os.environ.get("NCBI_API_KEY", "")
if api_key and not api_key.startswith("your_"):
    Entrez.api_key = api_key
else:
    Entrez.api_key = None

CHECKPOINT_FILE = "pubmed_checkpoint.json"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_checkpoint(processed_ids):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(processed_ids), f)

def search_pubmed(query, max_results=10000):
    print(f"Searching PubMed for: {query}")
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        print(f"Error during PubMed Search: {e}")
        return []

def fetch_abstracts(id_list):
    """Note: This is now integrated into the main loop for checkpointing"""
    pass

def main():
    parser = argparse.ArgumentParser(description="Massive PubMed Ingestion with Checkpointing.")
    parser.add_argument("--query", default="Chronic Metabolic Diseases", help="Search query")
    parser.add_argument("--limit", type=int, default=10000, help="Max abstracts to process")
    parser.add_argument("--batch", type=int, default=50, help="Batch size for checkpointing")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel extraction workers")
    
    args = parser.parse_args()

    # Init RAG
    print(f"⏳ Initializing RAG Backend (1B Model) with {args.workers} workers...")
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

    # 1. Search for total IDs
    all_ids = search_pubmed(args.query, max_results=args.limit)
    if not all_ids:
        print("❌ No matching papers found.")
        return

    # 2. Filter using checkpoint
    processed_ids = load_checkpoint()
    todo_ids = [pid for pid in all_ids if pid not in processed_ids]
    
    print(f"📊 Progress: {len(processed_ids)}/{len(all_ids)} completed.")
    print(f"🚀 Starting parallel ingestion for {len(todo_ids)} new papers...")

    # 3. Main Loop (Batched)
    batch_size = args.batch
    for i in range(0, len(todo_ids), batch_size):
        batch = todo_ids[i : i + batch_size]
        print(f"\n--- Batch {i//batch_size + 1}: {len(batch)} papers ---")
        
        try:
            # Fetch batch abstracts
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="abstract", retmode="text")
            data = handle.read()
            handle.close()
            
            abstracts = [p.strip() for p in data.split("\n\n") if len(p.strip()) > 100]
            
            # Process batch in parallel
            rag.load_reference_papers(abstracts, progress_callback=lambda msg: print(f"  {msg}"), max_workers=args.workers)
            
            # Update Checkpoint
            for pid in batch:
                processed_ids.add(pid)
            save_checkpoint(processed_ids)
            
            # Minimal rate limiting (Entrez rules)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"⚠️  Error in batch processing: {e}")
            print("💾 Checkpoint saved. You can restart the script to resume.")
            continue

    print("\n🎉 Massive Ingestion Complete!")

if __name__ == "__main__":
    main()

import os
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from api_clients import Neo4jClient
from llm_helpers import EmbeddingStore

def repair_layer(neo4j, emb_store, layer=None, batch_size=200, max_workers=4):
    layer_name = f"Layer {layer}" if layer else "All Layers"
    print(f"\n--- Starting parallel repair for {layer_name} with {max_workers} workers ---")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_batch():
        nodes = neo4j.get_unembedded_nodes(layer=layer, limit=batch_size)
        if not nodes:
            return False
            
        texts = [n.context for n in nodes]
        vectors = emb_store.embed_batch(texts)
        
        node_data = []
        for node, vec in zip(nodes, vectors):
            if vec is not None:
                node_data.append({
                    "name": node.name,
                    "embedding": vec.tolist()
                })
        
        if node_data:
            neo4j.batch_update_embeddings(node_data)
        return len(node_data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            # Submit a few batches at once to keep the pipeline full
            futures = [executor.submit(process_batch) for _ in range(max_workers)]
            results = [f.result() for f in as_completed(futures)]
            
            if not any(results):
                break
            
            total_batch = sum(r for r in results if isinstance(r, int))
            print(f"  Synced cumulative batch of {total_batch} entities...")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Repair missing embeddings in the Medical Graph.")
    parser.add_argument("--layer", type=int, help="Target specific layer (1, 2, or 3). If omitted, repairs all.")
    parser.add_argument("--batch", type=int, default=200, help="Batch size for embedding calculation.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    
    args = parser.parse_args()
    
    # Init Backend
    print("LOG: Initializing Embedding Model (CPU - all-MiniLM-L6-v2)...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    emb_store = EmbeddingStore(embedder)
    
    neo4j_creds = {
        "uri": os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "password")
    }
    
    neo4j = Neo4jClient(
        uri=neo4j_creds["uri"],
        user=neo4j_creds["user"],
        password=neo4j_creds["password"]
    )
    
    if args.layer:
        repair_layer(neo4j, emb_store, layer=args.layer, batch_size=args.batch, max_workers=args.workers)
    else:
        # Priority Repair: 1 -> 2 -> 3
        for L in [1, 2, 3]:
            repair_layer(neo4j, emb_store, layer=L, batch_size=args.batch, max_workers=args.workers)

    print("\nLog: Parallel Bridge Repair Session Complete.")

if __name__ == "__main__":
    main()

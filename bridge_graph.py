import os
import argparse
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from api_clients import Neo4jClient
from llm_helpers import EmbeddingStore

def bridge_layers(neo4j, emb_store, source_layer=2, target_layer=3, batch_size=50, max_workers=4):
    """
    Finds entities in source_layer (e.g., PubMed) and links them to target_layer (e.g., UMLS)
    based on vector similarity.
    """
    print(f"\n--- Bridging Layer {source_layer} -> Layer {target_layer} ({max_workers} workers) ---")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_batch():
        # 1. Get nodes that aren't linked yet
        nodes = neo4j.get_unlinked_entities(source_layer, target_layer, limit=batch_size)
        if not nodes:
            return False
            
        edge_data = []
        for s_ent in nodes:
            if s_ent.embedding is None:
                continue
                
            # 2. Find closest concepts in target layer using Neo4j Vector Search
            # We use the embedding we already have to find neighbors
            neighbors = neo4j.find_similar_entities(
                query_name=s_ent.name, 
                layer=target_layer, 
                limit=1, # Linking to the single best match for grounding
                vector=s_ent.embedding.tolist()
            )
            
            for t_ent in neighbors:
                # 3. Calculate exact similarity for the edge attribute
                # (Note: finder already used cosine, but we store it for the edge)
                sim = emb_store.similarity(s_ent.embedding, t_ent.name)
                
                if sim > 0.45: # SIMILARITY_THRESHOLD from MedGraphRAG
                    edge_data.append({
                        "source": s_ent.name,
                        "target": t_ent.name,
                        "type": "the_definition_of" if target_layer == 3 else "the_reference_of",
                        "similarity": sim
                    })
        
        # 4. Batch sync edges
        if edge_data:
            neo4j.sync_cross_layer_edges(edge_data)
            
        return len(edge_data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            futures = [executor.submit(process_batch) for _ in range(max_workers)]
            results = [f.result() for f in as_completed(futures)]
            
            if not any(results):
                print(f"DONE: No more unlinked entities found in Layer {source_layer}.")
                break
            
            total_edges = sum(r for r in results if isinstance(r, int))
            print(f"  Created {total_edges} cross-layer links...")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Bridge gaps between Graph Layers using Vector Similarity.")
    parser.add_argument("--source", type=int, default=2, help="Source layer (e.g. 2 for PubMed)")
    parser.add_argument("--target", type=int, default=3, help="Target layer (e.g. 3 for UMLS/Vocab)")
    parser.add_argument("--batch", type=int, default=50, help="Batch size for linking.")
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
    
    if not neo4j.driver:
        print("❌ Fatal: Could not connect to Neo4j.")
        return

    # Check if vector index exists
    neo4j.create_vector_index(dimensions=384) # MiniLM-L6-v2 is 384d
    
    bridge_layers(
        neo4j, emb_store, 
        source_layer=args.source, 
        target_layer=args.target, 
        batch_size=args.batch, 
        max_workers=args.workers
    )

    print("\nLog: Bridging Session Complete.")

if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv
from api_clients import Neo4jClient

def check_stats():
    load_dotenv()
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    
    client = Neo4jClient(uri, user, password)
    if not client.driver:
        print("❌ Could not connect to Neo4j.")
        return

    print("--- Neo4j Graph Statistics ---")
    for layer in [1, 2, 3]:
        count = client.get_layer_count(layer)
        print(f"Layer {layer}: {count} entities")
    
    with client.driver.session() as session:
        # Check relationships
        res = session.run("MATCH ()-[r]->() RETURN r.type as type, count(r) as c")
        print("\n--- Relationships ---")
        for record in res:
            print(f"{record['type']}: {record['c']}")
            
        # Check cross-layer links specifically
        res = session.run("MATCH (n:Entity)-[r:LINK]->(m:Entity) RETURN n.layer as src, m.layer as tgt, count(r) as c")
        print("\n--- Cross-Layer Links ---")
        for record in res:
            print(f"L{record['src']} -> L{record['tgt']}: {record['c']}")

    client.close()

if __name__ == "__main__":
    check_stats()

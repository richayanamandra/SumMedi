import os
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

def main():
    load_dotenv()
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")

    print(f"Connecting to Neo4j at {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n" + "="*40)
    print("      MEDGRAPHRAG LIVE MONITOR")
    print("="*40)
    print("Time | L1 Nodes | L2 Nodes | L3 Nodes | Total Edges")
    print("-" * 55)

    try:
        while True:
            with driver.session() as session:
                # Query for nodes in each layer
                node_query = (
                    "MATCH (n:Entity) "
                    "RETURN n.layer AS layer, count(n) AS count"
                )
                res = session.run(node_query)
                counts = {1: 0, 2: 0, 3: 0}
                for record in res:
                    layer = record["layer"]
                    if layer in counts:
                        counts[layer] = record["count"]

                # Query for total relationships
                rel_query = "MATCH ()-[r]->() RETURN count(r) AS count"
                res_rel = session.run(rel_query)
                rel_count = res_rel.single()["count"]

                timestamp = time.strftime("%H:%M:%S")
                print(f"{timestamp} | {counts[1]:<8} | {counts[2]:<8} | {counts[3]:<8} | {rel_count}")
                
            time.sleep(60)  # Update every 10 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        driver.close()

if __name__ == "__main__":
    main()

import os
import argparse
from umls_importer import load_umls_to_neo4j, load_umls_relationships_to_neo4j
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Seed Neo4j with UMLS data.")
    parser.add_argument("--conso", default="MRCONSO.RRF", help="Path to MRCONSO.RRF")
    parser.add_argument("--sty", default="MRSTY.RRF", help="Path to MRSTY.RRF")
    parser.add_argument("--rel", default="MRREL.RRF", help="Path to MRREL.RRF")
    parser.add_argument("--skip-nodes", action="store_true", help="Skip loading nodes")
    parser.add_argument("--skip-rels", action="store_true", help="Skip loading relationships")
    parser.add_argument("--clear", action="store_true", help="Clear L3 in Neo4j before seeding")

    args = parser.parse_args()
    load_dotenv()

    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")

    if args.clear:
        from api_clients import Neo4jClient
        client = Neo4jClient(uri, user, password)
        print("Clearing all L3 entities...")
        if client.driver:
            with client.driver.session() as session:
                session.run("MATCH (n:Entity) WHERE n.layer = 3 DETACH DELETE n")
        client.close()

    if not args.skip_nodes:
        print(f"🚀 Starting Node Import from {args.conso}...")
        load_umls_to_neo4j(args.conso, args.sty, uri, user, password)
        print("✅ Nodes imported.")

    if not args.skip_rels:
        print(f"🚀 Starting Relationship Import from {args.rel}...")
        load_umls_relationships_to_neo4j(args.conso, args.rel, uri, user, password)
        print("✅ Relationships imported.")

if __name__ == "__main__":
    main()

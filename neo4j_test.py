from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
pwd = os.getenv("NEO4J_PASSWORD", "password")

try:
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    print("✅ Neo4j Connection Successful")
except Exception as e:
    print(f"❌ Neo4j Connection Failed: {e}")

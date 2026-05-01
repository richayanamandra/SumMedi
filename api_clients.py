from typing import Optional
import requests
from neo4j import GraphDatabase
from data_models import Entity, Relationship

class UMLSClient:
    """
    Client for UMLS UTS REST API.
    Handles semantic search for terms and retrieval of definitions.
    """
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        # Simple in-memory cache: term -> {name, definition, type}
        self._cache: dict[str, dict] = {}

    def get_term_details(self, term: str) -> Optional[dict]:
        """
        Search for a term, find its CUI, and fetch definitions.
        Returns a dict with name, definition, and semantic type if found.
        """
        if not self.api_key:
            return None
        
        term_lower = term.lower()
        if term_lower in self._cache:
            return self._cache[term_lower]

        try:
            # 1. Search for Concept (CUI)
            search_url = f"{self.BASE_URL}/search/current"
            params = {
                "string": term,
                "apiKey": self.api_key,
                "searchType": "exact"  # better for specific medical terms
            }
            resp = requests.get(search_url, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("result", {}).get("results", [])

            if not results:
                # Try 'words' search if exact fails
                params["searchType"] = "words"
                resp = requests.get(search_url, params=params, timeout=10)
                results = resp.json().get("result", {}).get("results", [])

            if not results:
                return None

            # Pick the first result
            best_match = results[0]
            cui = best_match.get("ui")
            name = best_match.get("name")

            # 2. Get Definitions
            def_url = f"{self.BASE_URL}/content/current/CUI/{cui}/definitions"
            resp = requests.get(def_url, params={"apiKey": self.api_key}, timeout=10)
            
            definition = ""
            if resp.status_code == 200:
                defs = resp.json().get("result", [])
                if defs:
                    # Prefer NCI or MSH sources if available
                    for d in defs:
                        if d.get("rootSource") in ["NCI", "MSH"]:
                            definition = d.get("value")
                            break
                    if not definition:
                        definition = defs[0].get("value")

            # 3. Get Semantic Types (optional, but good for Entity object)
            # In UMLS API, we can get this from concept details
            concept_url = f"{self.BASE_URL}/content/current/CUI/{cui}"
            resp = requests.get(concept_url, params={"apiKey": self.api_key}, timeout=10)
            semantic_type = "Other"
            if resp.status_code == 200:
                stys = resp.json().get("result", {}).get("semanticTypes", [])
                if stys:
                    semantic_type = stys[0].get("name", "Other")

            if name and (definition or semantic_type):
                res = {
                    "name": name,
                    "definition": definition,
                    "type": semantic_type,
                    "source": "UMLS"
                }
                self._cache[term_lower] = res
                return res

        except Exception as e:
            print(f"UMLS API Error: {e}")
            return None

        return None


class Neo4jClient:
    """
    Client for syncing the knowledge graph to Neo4j.
    """
    def __init__(self, uri: Optional[str], user: Optional[str], password: Optional[str]):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        if uri and user and password:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self.driver.verify_connectivity()
            except Exception as e:
                print(f"Neo4j Connection Error: {e}")
                self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_db(self):
        if not self.driver: return
        with self.driver.session() as session:
            # Only delete L1 and L2 entities and their relationships
            session.run("MATCH (n:Entity) WHERE n.layer IN [1, 2] DETACH DELETE n")

    def clear_all_db(self):
        if not self.driver: return
        with self.driver.session() as session:
            # Delete every entity across all layers
            session.run("MATCH (n:Entity) DETACH DELETE n")

    def clear_all_relationships(self):
        if not self.driver: return
        with self.driver.session() as session:
            # Batch delete relationships to prevent memory crash on 27M edges
            while True:
                res = session.run("MATCH ()-[r]->() WITH r LIMIT 50000 DELETE r RETURN count(r) AS deleted_count")
                record = res.single()
                if not record or record["deleted_count"] == 0:
                    break

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        if not self.driver: return None
        with self.driver.session() as session:
            res = session.run("MATCH (n:Entity {name: $name}) RETURN n", name=name)
            record = res.single()
            if record:
                n = record["n"]
                return Entity(
                    name=n["name"],
                    entity_type=n["type"],
                    context=n["context"],
                    definition=n.get("definition", ""),
                    layer=n["layer"]
                )
        return None

    def find_similar_entities(self, query_name: str, layer: int, limit: int = 5, vector: list[float] = None) -> list[Entity]:
        """
        Hyper-scalable search: Uses Vector Index if available, 
        otherwise falls back gracefully to fuzzy name search.
        """
        if not self.driver: return []
        with self.driver.session() as session:
            res = None
            if vector and len(vector) > 0:
                # Optimized Vector Search
                query = (
                    "CALL db.index.vector.queryNodes('entity_embeddings', $limit, $vector) "
                    "YIELD node AS n, score "
                    "WHERE n.layer = $layer "
                    "RETURN n, score"
                )
                try:
                    res = session.run(query, vector=vector, layer=layer, limit=limit)
                    # We must consume/check at least one record to see if it failed
                    entities = []
                    for record in res:
                        n = record["n"]
                        entities.append(Entity(
                            name=n["name"],
                            entity_type=n.get("type", "Unknown"),
                            context=n.get("context", ""),
                            definition=n.get("definition", ""),
                            layer=n["layer"]
                        ))
                    return entities
                except Exception as e:
                    # Fallback if index is not created yet
                    if "no such vector schema index" in str(e).lower():
                        pass 
                    else:
                        print(f"Vector search warning: {e}")

            # Fallback to case-insensitive partial match
            query = (
                "MATCH (n:Entity) "
                "WHERE n.layer = $layer AND n.name =~('(?i).*'+$name+'.*') "
                "RETURN n LIMIT $limit"
            )
            res = session.run(query, name=query_name, layer=layer, limit=limit)
            entities = []
            for record in res:
                n = record["n"]
                entities.append(Entity(
                    name=n["name"],
                    entity_type=n.get("type", "Unknown"),
                    context=n.get("context", ""),
                    definition=n.get("definition", ""),
                    layer=n["layer"]
                ))
            return entities

    def create_vector_index(self, dimensions: int = 384):
        """
        Initialize the Neo4j Vector Index for medical entities.
        """
        if not self.driver: return
        with self.driver.session() as session:
            # Create vector index if not exists (Neo4j 5.15+ syntax)
            query = f"""
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (n:Entity) ON (n.embedding)
            OPTIONS {{indexConfig: {{
             `vector.dimensions`: {dimensions},
             `vector.similarity_function`: 'cosine'
            }}}}
            """
            try:
                session.run(query)
                print("✅ Neo4j Vector Index 'entity_embeddings' created/verified.")
            except Exception as e:
                print(f"Error creating Vector Index: {e}")

    def get_neighbors(self, entity_name: str, hops: int = 2) -> list[Entity]:
        if not self.driver: return []
        with self.driver.session() as session:
            query = (
                "MATCH (s:Entity {name: $name})-[*1.."+str(hops)+"]-(t:Entity) "
                "RETURN DISTINCT t"
            )
            res = session.run(query, name=entity_name)
            entities = []
            for record in res:
                n = record["t"]
                entities.append(Entity(
                    name=n["name"],
                    entity_type=n.get("type", "Unknown"),
                    context=n.get("context", ""),
                    definition=n.get("definition", ""),
                    layer=n.get("layer", 1)
                ))
            return entities

    def get_layer_count(self, layer: int) -> int:
        if not self.driver: return 0
        with self.driver.session() as session:
            res = session.run("MATCH (n:Entity) WHERE n.layer = $layer RETURN count(n) AS c", layer=layer)
            record = res.single()
            return record["c"] if record else 0

    def sync_entities(self, entities: list[Entity]):
        if not self.driver: return
        with self.driver.session() as session:
            # Batch MERGE for efficiency
            query = (
                "UNWIND $batch AS item "
                "MERGE (n:Entity {name: item.name}) "
                "SET n.type = item.type, n.context = item.context, "
                "    n.definition = item.definition, n.layer = item.layer "
                "WITH n, item "
                "CALL apoc.create.addLabels(n, [item.label]) YIELD node "
                "RETURN count(*)"
            )
            # fallback if APOC is not installed
            query_simple = (
                "UNWIND $batch AS item "
                "MERGE (n:Entity {name: item.name}) "
                "SET n.type = item.type, n.context = item.context, "
                "    n.definition = item.definition, n.layer = item.layer, "
                "    n.layer_label = item.label "
                "RETURN count(*)"
            )
            
            batch = [
                {
                    "name": e.name, 
                    "type": e.entity_type, 
                    "context": e.context, 
                    "definition": e.definition, 
                    "layer": e.layer,
                    "label": f"L{e.layer}_Entity"
                } for e in entities
            ]
            try:
                session.run(query_simple, batch=batch)
            except Exception as e:
                print(f"Neo4j Sync Entities Error: {e}")

    def sync_relationships(self, relationships: list[Relationship]):
        if not self.driver: return
        with self.driver.session() as session:
            query = (
                "UNWIND $batch AS item "
                "MATCH (s:Entity {name: item.source}), (t:Entity {name: item.target}) "
                "MERGE (s)-[rel:RELATED_TO {type: item.rel_type}]->(t) "
                "RETURN count(*)"
            )
            batch = [
                {
                    "source": r.source, 
                    "target": r.target, 
                    "rel_type": r.relation
                } for r in relationships
            ]
            try:
                session.run(query, batch=batch)
            except Exception as e:
                print(f"Neo4j Sync Relationships Error: {e}")

    def add_cross_layer_edge(self, source_name: str, target_name: str, relation_type: str, similarity: float):
        if not self.driver: return
        with self.driver.session() as session:
            try:
                session.run(
                    "MATCH (s:Entity {name: $source}), (t:Entity {name: $target}) "
                    "MERGE (s)-[rel:LINK {type: $rel_type}]->(t) "
                    "SET rel.similarity = $sim",
                    source=source_name, target=target_name, rel_type=relation_type, sim=similarity
                )
            except Exception as e:
                print(f"Neo4j Sync Cross-Layer Edge Error: {e}")
    def get_unembedded_nodes(self, layer: int = None, limit: int = 100) -> list[Entity]:
        """
        Retrieves nodes that are missing vector embeddings.
        """
        if not self.driver: return []
        with self.driver.session() as session:
            where_clause = "WHERE n.embedding IS NULL"
            if layer is not None:
                where_clause += f" AND n.layer = {layer}"
            
            query = f"MATCH (n:Entity) {where_clause} RETURN n LIMIT $limit"
            res = session.run(query, limit=limit)
            entities = []
            for record in res:
                n = record["n"]
                entities.append(Entity(
                    name=n["name"],
                    entity_type=n.get("type", "Unknown"),
                    context=n.get("context", ""),
                    definition=n.get("definition", ""),
                    layer=n["layer"]
                ))
            return entities

    def batch_update_embeddings(self, node_data: list[dict]):
        """
        High-speed batch update for embeddings.
        node_data: list of {'name': str, 'embedding': list[float]}
        """
        if not self.driver or not node_data: return
        with self.driver.session() as session:
            query = (
                "UNWIND $batch AS item "
                "MATCH (n:Entity {name: item.name}) "
                "SET n.embedding = item.embedding "
                "RETURN count(*)"
            )
            try:
                session.run(query, batch=node_data)
            except Exception as e:
                print(f"Batch Update Embeddings Error: {e}")
    def get_unlinked_entities(self, source_layer: int, target_layer: int, limit: int = 100) -> list[Entity]:
        """
        Finds nodes in source_layer that do NOT have a LINK to target_layer.
        Useful for bridging gaps between PubMed (L2) and UMLS (L3).
        """
        if not self.driver: return []
        with self.driver.session() as session:
            # Note: We ensure they HAVE an embedding before trying to link them
            query = (
                "MATCH (n:Entity {layer: $s_layer}) "
                "WHERE n.embedding IS NOT NULL "
                "AND NOT (n)-[:LINK]->(:Entity {layer: $t_layer}) "
                "RETURN n LIMIT $limit"
            )
            res = session.run(query, s_layer=source_layer, t_layer=target_layer, limit=limit)
            entities = []
            for record in res:
                n = record["n"]
                import numpy as np
                entities.append(Entity(
                    name=n["name"],
                    entity_type=n.get("type", "Unknown"),
                    context=n.get("context", ""),
                    definition=n.get("definition", ""),
                    layer=n["layer"],
                    embedding=np.array(n["embedding"], dtype=np.float32) if n.get("embedding") else None
                ))
            return entities

    def sync_cross_layer_edges(self, edge_data: list[dict]):
        """
        Batch MERGE for cross-layer edges.
        edge_data: list of {'source': str, 'target': str, 'type': str, 'similarity': float}
        """
        if not self.driver or not edge_data: return
        with self.driver.session() as session:
            query = (
                "UNWIND $batch AS item "
                "MATCH (s:Entity {name: item.source}), (t:Entity {name: item.target}) "
                "MERGE (s)-[rel:LINK {type: item.type}]->(t) "
                "SET rel.similarity = item.similarity "
                "RETURN count(*)"
            )
            try:
                session.run(query, batch=edge_data)
            except Exception as e:
                print(f"Batch Sync Cross-Layer Edges Error: {e}")

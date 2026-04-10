import os
import json
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j+s://92fb1806.databases.neo4j.io")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j") 
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

# LLM Configuration (Gemini)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Load the same model used for uploading
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_graph_context(query_text, top_k=10):
    query_vector = model.encode(query_text).tolist()
    context_triples = []
    
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            cypher = """
            CALL db.index.vector.queryNodes('entity_embeddings', $k, $vector)
            YIELD node, score
            MATCH (node)-[r]->(neighbor)
            RETURN 
                node.id AS source, 
                node.label AS source_label, 
                type(r) AS rel_type, 
                r.label AS rel_label,
                neighbor.id AS target, 
                neighbor.label AS target_label,
                score
            LIMIT 50
            """
            result = session.run(cypher, vector=query_vector, k=top_k)
            for record in result:
                triple = (
                    f"({record['source']}:{record['source_label']}) "
                    f"--[{record['rel_label'] or record['rel_type']}]--> "
                    f"({record['target']}:{record['target_label']})"
                )
                context_triples.append(triple)
    return "\n".join(context_triples)

def query_llm(prompt, context):
    full_prompt = f"""
You are a helpful assistant. Answer the user's question based ONLY on the following knowledge graph context.
If the context doesn't contain the answer, say you don't know.

### KNOWLEDGE GRAPH CONTEXT:
{context}

### USER QUESTION:
{prompt}

### ANSWER:
"""
    if not GOOGLE_API_KEY:
        return "ERROR: No GOOGLE_API_KEY found in .env"

    try:
        # gemini-3-flash-preview is the standard naming convention
        model = genai.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini: {e}"

def main():
    if not NEO4J_PASSWORD:
        print("Error: NEO4J_PASSWORD environment variable not set.")
        return

    print("\n--- GraphRAG Terminal Interface (Gemini Powered) ---")
    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ").strip()
        if user_query.lower() in ['exit', 'quit']:
            break
        if not user_query:
            continue

        print("\n1. Searching graph for context...")
        context = get_graph_context(user_query)
        
        if not context or context == "No relevant nodes found in the graph.":
            print("No relevant graph context found.")
        else:
            print(f"Found {len(context.splitlines())} relevant triples:")
            for line in context.splitlines():
                print(f"  -> {line}")

        print("2. Querying Gemini...")
        answer = query_llm(user_query, context)
        
        print("\n=== GEMINI RESPONSE ===")
        print(answer)
        print("====================\n")

if __name__ == "__main__":
    main()

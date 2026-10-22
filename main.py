import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="Cloud Vector RAG Engine")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

class IngestRequest(BaseModel):
    text: str
    source: str = "knowledge-base"

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {"status": "active", "service": "Cloud Vector RAG API Engine"}

@app.post("/ingest")
async def ingest_document(payload: IngestRequest):
    """Generates a text embedding using gemini-embedding-001 and saves text to Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="Supabase environment variables missing.")

    # 1. Generate text embedding via Gemini using gemini-embedding-001
    try:
        embed_response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=payload.text
        )
        embedding = embed_response['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Embedding failed: {str(e)}")

    # 2. Store content + embedding vector in Supabase via REST API
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    db_payload = {
        "content": payload.text,
        "embedding": embedding,
        "metadata": {"source": payload.source}
    }
    
    endpoint = f"{SUPABASE_URL}/rest/v1/document_knowledge"
    res = requests.post(endpoint, json=db_payload, headers=headers)
    
    if res.status_code not in [200, 201]:
        raise HTTPException(status_code=500, detail=f"Database write failed: {res.text}")
        
    return {"status": "success", "message": "Text embedding stored in vector DB"}

@app.post("/query")
async def query_knowledge_base(payload: QueryRequest):
    """Executes vector similarity search and answers using Gemini RAG."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="Supabase environment variables missing.")

    # 1. Embed incoming question
    try:
        query_embed = genai.embed_content(
            model="models/gemini-embedding-001",
            content=payload.question
        )['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query embedding failed: {str(e)}")

    # 2. Run vector cosine search RPC query in Supabase
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    
    rpc_payload = {
        "query_embedding": query_embed,
        "match_threshold": 0.2,
        "match_count": 3
    }
    
    endpoint = f"{SUPABASE_URL}/rest/v1/rpc/match_documents"
    rpc_res = requests.post(endpoint, json=rpc_payload, headers=headers)
    
    if rpc_res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {rpc_res.text}")
        
    matches = rpc_res.json()
    retrieved_chunks = [item['content'] for item in matches] if matches else []
    context_str = "\n---\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."

    # 3. Construct RAG prompt and answer
    prompt = (
        "You are an expert AI knowledge assistant. Answer the user's question relying strictly on the retrieved database context below.\n"
        "If the answer is not contained in the context, clearly state that the information is unavailable in the database.\n\n"
        f"Retrieved Context:\n{context_str}\n\n"
        f"User Question: {payload.question}"
    )
    
    answer = model.generate_content(prompt)
    
    return {
        "answer": answer.text,
        "context_used": retrieved_chunks
    }
    

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv
from groq import Groq
from upstash_redis import Redis
import PyPDF2
from io import BytesIO
from typing import Optional, List
from uuid import uuid4
from jose import jwt, JWTError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Upstash Redis using REST API
redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

TTL_SECONDS = 86400  # 24 hours

class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"  # Default session if not provided
    document_id: Optional[str] = None

class ClearChat(BaseModel):
    session_id: str = "default"

SECRET_KEY = "yaFAWKKMmjwYG9YU5fHxmnz4xDecCrOmlTSlkXu47eM="  # Use a secure random string!
ALGORITHM = "HS256"

def create_jwt(user_id: str):
    return jwt.encode({"sub": user_id}, SECRET_KEY, algorithm=ALGORITHM)

def get_user_id_from_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None

@app.get("/")
async def root():
    return {
        "message": "Chatbot API is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/get-token")
def get_token():
    user_id = str(uuid4())
    token = create_jwt(user_id)
    return {"token": token}

def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = auth_header.split(" ")[1]
    user_id = get_user_id_from_jwt(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id

def get_conversation(session_id: str):
    data = redis.get(f"chat:{session_id}")
    return json.loads(data) if data else []


def save_conversation(session_id: str, messages: list):
    key = f"chat:{session_id}"
    if redis.exists(key):
        redis.set(key,json.dumps(messages), keepttl=True)
    else:
        redis.setex(key, TTL_SECONDS, json.dumps(messages))

def extract_text_from_pdf(file_bytes):
    """Extract text from pdf"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words),chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document for RAG"""
    try:
        #Read file
        file_bytes = await file.read()
        #Extract text
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
        elif file.filename.endswith('.txt'):
            text = file_bytes.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files supported")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in document")
            
        #Split into chunks
        chunks = chunk_text(text)

        #Creat document id
        doc_id = f"{file.filename.replace(' ', '_').replace('.', '_')}_{uuid4().hex}"
        # Store all chunks for this document in Redis as a JSON list
        redis.set(f"chunks:{doc_id}", json.dumps(chunks))
        doc_info = {
            "id" : doc_id,
            "filename" : file.filename,
            "chunks_count" : len(chunks),
            "upload_date" : str(os.times())
        }
        redis.set(f"doc:{doc_id}",json.dumps(doc_info))

        #Add to document list 
        doc_list = redis.get("document_list")
        doc_list = json.loads(doc_list) if doc_list else []
        if doc_id not in doc_list:
            doc_list.append(doc_id)
        redis.set("document_list", json.dumps(doc_list))

        return {
            "message" : "Document uploaded successfully",
            "document_id": doc_id,
            "filename" : file.filename,
            "chunks" : len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        doc_list = redis.get("document_list")
        doc_ids = json.loads(doc_list) if doc_list else []

        documents = []
        for doc_id in doc_ids:
            doc_data = redis.get(f"doc:{doc_id}")
            if doc_data:
                documents.append(json.loads(doc_data))
        return {"documents" : documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(data: ChatMessage, user_id: str = Depends(get_current_user)):
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    # Get existing conversation
    messages = get_conversation(user_id)
    
    try:
        #document_id provided, do rag search
        context = None
        if data.document_id:
            #create embedding for the question 
            # Load chunks for the requested document
            chunks_json = redis.get(f"chunks:{data.document_id}")
            if not chunks_json:
                context = None
            else:
                chunks = json.loads(chunks_json)
                if not chunks:
                    context = None
                else:
                    # Compute TF-IDF on document chunks and the question
                    vectorizer = TfidfVectorizer().fit(chunks)
                    chunks_matrix = vectorizer.transform(chunks)             # shape: (n_chunks, n_features)
                    query_vec = vectorizer.transform([data.message])         # shape: (1, n_features)
                    # cosine similarity between query and each chunk
                    sims = cosine_similarity(query_vec, chunks_matrix).flatten()
                    # pick top-k most similar chunks
                    top_k = 3
                    top_indices = np.argsort(sims)[-top_k:][::-1]
                    retrieved_chunks = [chunks[i] for i in top_indices if sims[i] > 0]
                    if retrieved_chunks:
                        context = "\n\n".join(retrieved_chunks)
                    else:
                        context = None

        #build messages for groq
        groq_messages = []
        
        groq_messages.append({
            "role": "system",
            "content": "Use the history of the conversation as facts to help you answer this question."
        })
        #add system message with contyext if rag is used
        if context:
            groq_messages.append({
                "role" : "system",
                "content": f"you are a helpful assistant. Anser thus user's question based on the following context from their document:\n\n{context}\n\nif the answer is not in the context, say so politely."
            })
            #add conversation history
            groq_messages.extend(messages)

            #add current user message
            groq_messages.append({"role": "user","content": data.message})

            #Call groq
            client = Groq(api_key = api_key)
            response = client.chat.completions.create(
                model = "llama-3.3-70b-versatile",
                messages = groq_messages,
                max_tokens = 500,
            )

            assistant_message = response.choices[0].message.content

            #sava to conversation history (without system message)
            messages.append({"role":"user","content": data.message})
            messages.append({"role": "assistant", "content": assistant_message})
            save_conversation(user_id, messages)

            return {
                "response": assistant_message,
                "history": messages,
                "used_rag": bool(context)
            }
        else:
            #add conversation history
            groq_messages.extend(messages)

            #add current user message
            groq_messages.append({"role": "user","content": data.message})

            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=groq_messages,
                max_tokens=500,
            )
            assistant_message = response.choices[0].message.content
            messages.append({"role": "user", "content": data.message})
            messages.append({"role": "assistant", "content": assistant_message})
            save_conversation(user_id, messages)
            # Always return a response here!
            return {
                "response": assistant_message,
                "history": messages,
                "used_rag": False
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        # Get document info
        doc_data = redis.get(f"doc:{document_id}")
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_info = json.loads(doc_data)
        
        # Delete vectors from Upstash Vector
        # Note: Upstash Vector doesn't have bulk delete by prefix yet
        # So we delete each chunk individually
        # Delete stored chunks from Redis
        redis.delete(f"chunks:{document_id}")
        
        # Delete from Redis
        redis.delete(f"doc:{document_id}")
        
        # Remove from document list
        doc_list = redis.get("document_list")
        doc_list = json.loads(doc_list) if doc_list else []
        if document_id in doc_list:
            doc_list.remove(document_id)
        if doc_list:
            redis.set("document_list", json.dumps(doc_list))
        else:
            redis.delete("document_list")  # Delete the key if empty

        
        return {"message": "Document deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_conversation(user_id: str = Depends(get_current_user)):
    """Clear conversation history for a session"""
    redis.delete(f"chat:{user_id}")
    return {"message": "Conversation cleared", "session_id": user_id}


@app.get("/history")
async def get_history(user_id: str = Depends(get_current_user)):
    """Get conversation history for a session"""
    return {
        "history": get_conversation(user_id)
    }


@app.get("/health")
async def health():
    return {"status": "ok"}

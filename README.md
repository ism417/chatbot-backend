# chatbot-backend

A FastAPI-based chatbot backend with RAG (Retrieval-Augmented Generation) capabilities, powered by Groq's LLaMA 3.3 70B model and Upstash Redis for conversation persistence.

## Features

- 🤖 **AI-Powered Chat** - Uses Groq's LLaMA 3.3 70B model for intelligent responses
- 📄 **Document Upload** - Upload PDF and TXT files for context-aware conversations
- 🔍 **RAG Search** - TF-IDF based retrieval for relevant document chunks
- 💾 **Conversation History** - Persistent chat history stored in Upstash Redis (24-hour TTL)
- 🔐 **JWT Authentication** - Secure token-based user identification
- 🌐 **CORS Enabled** - Ready for frontend integration

## Tech Stack

- **Framework**: FastAPI
- **AI Model**: Groq (LLaMA 3.3 70B Versatile)
- **Database**: Upstash Redis
- **PDF Processing**: PyPDF2
- **Text Search**: scikit-learn (TF-IDF + Cosine Similarity)
- **Authentication**: python-jose (JWT)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatbot-backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   UPSTASH_REDIS_REST_URL=your_upstash_redis_url
   UPSTASH_REDIS_REST_TOKEN=your_upstash_redis_token
   ```

4. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

### `GET /`

Returns basic API information and available routes.

**Response:**
```json
{
  "message": "Chatbot API is running",
  "docs": "/docs",
  "health": "/health"
}
```

---

### `GET /health`

Health check endpoint to verify the API is running.

**Response:**
```json
{
  "status": "ok"
}
```

---

### `GET /get-token`

Generates a new JWT token for authentication. Each token is associated with a unique user ID.

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

---

### `POST /chat`

Send a message to the chatbot. Supports optional RAG (Retrieval-Augmented Generation) when a document ID is provided.

**Authentication:** Required (Bearer Token)

**Headers:**
```
Authorization: Bearer <your_token>
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | ✅ | The user's message to the chatbot |
| `session_id` | string | ❌ | Session identifier (default: "default") |
| `document_id` | string | ❌ | Document ID for RAG-based responses |

**Example Request:**
```json
{
  "message": "What is machine learning?",
  "session_id": "default",
  "document_id": "my_document_pdf_abc123"
}
```

**Response:**
```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "history": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ],
  "used_rag": true
}
```

| Field | Description |
|-------|-------------|
| `response` | The chatbot's reply |
| `history` | Full conversation history |
| `used_rag` | Whether document context was used |

---

### `POST /clear`

Clears the conversation history for the authenticated user.

**Authentication:** Required (Bearer Token)

**Headers:**
```
Authorization: Bearer <your_token>
```

**Response:**
```json
{
  "message": "Conversation cleared",
  "session_id": "user-uuid-here"
}
```

---

### `GET /history`

Retrieves the conversation history for the authenticated user.

**Authentication:** Required (Bearer Token)

**Headers:**
```
Authorization: Bearer <your_token>
```

**Response:**
```json
{
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help you today?"}
  ]
}
```

---

### `POST /upload`

Upload a document (PDF or TXT) for RAG-based conversations. The document is split into chunks and stored in Redis.

**Content-Type:** `multipart/form-data`

**Supported Formats:** `.pdf`, `.txt`

**Request:**
| Field | Type | Description |
|-------|------|-------------|
| `file` | File | The document file to upload |

**Response:**
```json
{
  "message": "Document uploaded successfully",
  "document_id": "my_document_pdf_abc123def456",
  "filename": "my_document.pdf",
  "chunks": 15
}
```

| Field | Description |
|-------|-------------|
| `document_id` | Unique ID to reference this document in chat |
| `filename` | Original filename |
| `chunks` | Number of text chunks created |

**Errors:**
- `400` - Only PDF and TXT files supported
- `400` - No text found in document
- `500` - Server error

---

### `GET /documents`

Lists all uploaded documents.

**Response:**
```json
{
  "documents": [
    {
      "id": "my_document_pdf_abc123",
      "filename": "my_document.pdf",
      "chunks_count": 15,
      "upload_date": "..."
    }
  ]
}
```

---

### `DELETE /document/{document_id}`

Deletes a specific document and its associated chunks from the system.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `document_id` | string | The ID of the document to delete |

**Response:**
```json
{
  "message": "Document deleted successfully"
}
```

**Errors:**
- `404` - Document not found
- `500` - Server error

---

## Usage Examples

### Basic Chat Flow

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Get authentication token
token_response = requests.get(f"{BASE_URL}/get-token")
token = token_response.json()["token"]

headers = {"Authorization": f"Bearer {token}"}

# 2. Send a chat message
chat_response = requests.post(
    f"{BASE_URL}/chat",
    headers=headers,
    json={"message": "Hello, how are you?"}
)
print(chat_response.json()["response"])

# 3. Get conversation history
history = requests.get(f"{BASE_URL}/history", headers=headers)
print(history.json())

# 4. Clear conversation
requests.post(f"{BASE_URL}/clear", headers=headers)
```

### RAG Chat with Document

```python
# 1. Upload a document
with open("document.pdf", "rb") as f:
    upload_response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": f}
    )
document_id = upload_response.json()["document_id"]

# 2. Chat with document context
chat_response = requests.post(
    f"{BASE_URL}/chat",
    headers=headers,
    json={
        "message": "Summarize the main points",
        "document_id": document_id
    }
)
print(chat_response.json()["response"])

# 3. Delete document when done
requests.delete(f"{BASE_URL}/document/{document_id}")
```

## Configuration

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | API key for Groq |
| `UPSTASH_REDIS_REST_URL` | Upstash Redis REST URL |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash Redis REST Token |

## Notes

- Conversation history expires after 24 hours (configurable via `TTL_SECONDS`)
- Document chunks use a default size of 500 words with 50-word overlap
- RAG retrieval returns top 3 most relevant chunks using TF-IDF cosine similarity
- Maximum response tokens: 500

## License

MIT

A FastAPI-based chatbot backend with RAG (Retrieval-Augmented Generation) capabilities, powered by Groq's LLaMA 3.3 70B model and Upstash Redis for conversation persistence.

## Features

- 🤖 **AI-Powered Chat** - Uses Groq's LLaMA 3.3 70B model for intelligent responses
- 📄 **Document Upload** - Upload PDF and TXT files for context-aware conversations
- 🔍 **RAG Search** - TF-IDF based retrieval for relevant document chunks
- 💾 **Conversation History** - Persistent chat history stored in Upstash Redi (24-hour TTL)
- 🔐 **JWT Authentication** - Secure token-based user identification
- 🌐 **CORS Enabled** - Ready for frontend integration

## Tech Stack

- **Framework**: FastAPI
- **AI Model**: Groq (LLaMA 3.3 70B Versatile)
- **Database**: Upstash Redis
- **PDF Processing**: PyPDF2
- **Text Search**: scikit-learn (TF-IDF + Cosine Similarity)
- **Authentication**: python-jose (JWT)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatbot-backend
   ```

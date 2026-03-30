import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel

import config
from rag_chat import generate_chat_answer

load_dotenv()


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None


app = FastAPI(title="Medical Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache
def get_hf_token() -> str:
    return _require_env("HF_TOKEN")


@lru_cache
def get_hf_client() -> InferenceClient:
    return InferenceClient(token=get_hf_token(), timeout=120)


@lru_cache
def get_index() -> faiss.Index:
    return faiss.read_index(str(Path(config.DB_FAISS_PATH) / "index.faiss"))


@lru_cache
def get_documents() -> List[Dict[str, object]]:
    path = Path(config.DB_FAISS_PATH) / "documents.json"
    return json.loads(path.read_text(encoding="utf-8"))


def embed_query(text: str) -> np.ndarray:
    vector = get_hf_client().feature_extraction(text=text, model=config.EMBEDDING_MODEL)
    values = np.asarray(vector, dtype="float32")
    if values.ndim == 2:
        values = values.mean(axis=0)
    return values.reshape(1, -1)


def retrieve_documents(prompt: str) -> List[Dict[str, object]]:
    scores, indices = get_index().search(embed_query(prompt), config.RETRIEVAL_K)
    documents = get_documents()
    results: List[Dict[str, object]] = []

    for idx in indices[0]:
        if idx < 0 or idx >= len(documents):
            continue
        results.append(documents[idx])

    return results


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Medical chatbot API is running."}


@app.get("/health")
def health() -> Dict[str, str]:
    try:
        get_index()
        get_documents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, object]:
    query = (request.query or request.message or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Provide `query` or `message`.")

    try:
        documents = retrieve_documents(query)
        answer = generate_chat_answer(
            prompt=query,
            source_documents=documents,
            client=get_hf_client(),
            model_name=config.HUGGINGFACE_REPO_ID,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "answer": answer,
        "sources": [doc.get("metadata", {}).get("source", "") for doc in documents],
    }

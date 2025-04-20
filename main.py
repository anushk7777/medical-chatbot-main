import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings

# Load env vars
load_dotenv()

app = FastAPI()

# Enable CORS so your Lovable frontend can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    query: str

# Custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Load LLM and vectorstore
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss/"

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task='text-generation',
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = qa_chain.invoke({"query": request.query})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "") for doc in result["source_documents"]]
        }
    except Exception as e:
        return {"error": str(e)}

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Importazioni dirette per bypassare il modulo 'langchain'
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain
    if "GOOGLE_API_KEY" not in os.environ: return
    try:
        all_documents = []
        if os.path.exists("data_pdfs"): all_documents.extend(PyPDFDirectoryLoader("data_pdfs").load())
        web_loader = WebBaseLoader(["https://app.aipermind.com/strategy-lab", "https://app.aipermind.com/faq"])
        all_documents.extend(web_loader.load())

        if all_documents:
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_documents)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
            prompt = ChatPromptTemplate.from_messages([("system", "Contesto: {context}"), ("human", "{input}")])
            
            # Qui usiamo le funzioni direttamente
            rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    except Exception as e: logger.error(f"Errore: {e}")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if rag_chain is None: raise HTTPException(status_code=503, detail="Non pronto")
    return {"reply": rag_chain.invoke({"input": request.message})["answer"]}

@app.get("/")
def health_check(): return {"status": "Operativo"}

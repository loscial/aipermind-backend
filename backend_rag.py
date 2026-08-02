from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging

# --- Import moderni per LangChain ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS per frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

rag_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain
    
    if "GOOGLE_API_KEY" not in os.environ:
        logger.error("ERRORE CRITICO: GOOGLE_API_KEY mancante!")
        return
        
    try:
        all_documents = []
        PDF_DIRECTORY = "data_pdfs"

        # 1. Caricamento PDF
        if os.path.exists(PDF_DIRECTORY):
            pdf_loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
            all_documents.extend(pdf_loader.load())
        else:
            logger.warning(f"Cartella {PDF_DIRECTORY} non trovata.")

        # 2. Caricamento Web
        URLS = ["https://app.aipermind.com/strategy-lab", "https://app.aipermind.com/faq"]
        web_loader = WebBaseLoader(URLS)
        all_documents.extend(web_loader.load())

        if all_documents:
            # Chunking leggero
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_documents)
            
            # Embeddings via Google (non consumano RAM sul server)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            # Modello LLM
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
            
            # Prompt professionale
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Sei l'assistente ufficiale di aipermind lab. Rispondi basandoti esclusivamente sul seguente contesto:\n\n{context}"),
                ("human", "{input}"),
            ])
            
            # Catena RAG
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
            logger.info("✅ RAG Inizializzato con successo.")
        else:
            logger.error("Nessun documento trovato per il RAG.")
            
    except Exception as e:
        logger.error(f"❌ Errore critico: {e}")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Sistema non ancora pronto.")
    try:
        response = rag_chain.invoke({"input": request.message})
        return {"reply": response["answer"]}
    except Exception as e:
        logger.error(f"Errore generazione: {e}")
        raise HTTPException(status_code=500, detail="Errore elaborazione.")

@app.get("/")
def health_check():
    return {"status": "Operativo"}

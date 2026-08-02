from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging

# LangChain Imports per RAG Multi-sorgente
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Configura il logging per vedere cosa succede nel terminale di Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- CORS: Fondamentale per far comunicare il sito (frontend) con questo server ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In produzione potresti mettere l'URL esatto del tuo sito
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Variabile globale per mantenere in memoria il sistema RAG
rag_chain = None

# --- CONFIGURAZIONE SORGENTI DATI ---
# 1. Cartella locale dove metteremo i PDF (questa cartella deve esistere su GitHub)
PDF_DIRECTORY = "data_pdfs"

# 2. Lista di siti web da leggere
URLS_TO_SCRAPE = [
    "https://app.aipermind.com/strategy-lab",
    "https://app.aipermind.com/faq"
    # Aggiungi qui altri URL se necessario
]

@app.on_event("startup")
async def startup_event():
    global rag_chain
    
    # Controllo di sicurezza vitale: senza API Key non andiamo da nessuna parte
    if "GOOGLE_API_KEY" not in os.environ:
        logger.error("ERRORE CRITICO: GOOGLE_API_KEY non trovata nelle variabili d'ambiente di Render!")
        return
        
    logger.info("Avvio caricamento dati. Ottimizzazione memoria RAM attiva (Google Embeddings)...")

    try:
        all_documents = []

        # --- FASE 1A: Caricamento PDF dalla cartella locale ---
        if not os.path.exists(PDF_DIRECTORY):
            os.makedirs(PDF_DIRECTORY)
            logger.warning(f"Cartella '{PDF_DIRECTORY}' creata ora. È vuota, non ci sono PDF da leggere.")
        else:
            logger.info(f"Lettura dei PDF dalla cartella '{PDF_DIRECTORY}' in corso...")
            try:
                pdf_loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
                pdf_docs = pdf_loader.load()
                all_documents.extend(pdf_docs)
                logger.info(f"Fatto! Trovati e caricati {len(pdf_docs)} frammenti dai PDF.")
            except Exception as e:
                logger.error(f"Errore durante la lettura dei PDF: {e}")

        # --- FASE 1B: Caricamento contenuti dai siti web ---
        if URLS_TO_SCRAPE:
            logger.info(f"Lettura dei siti web in corso...")
            try:
                web_loader = WebBaseLoader(URLS_TO_SCRAPE)
                web_docs = web_loader.load()
                all_documents.extend(web_docs)
                logger.info(f"Fatto! Caricati {len(web_docs)} documenti dal web.")
            except Exception as e:
                logger.error(f"Errore durante il caricamento dei siti web: {e}")

        if not all_documents:
            logger.warning("ATTENZIONE: Nessun documento trovato. Il Chatbot risponderà ma non avrà contesto sui tuoi file.")
            # Non blocchiamo l'app, ma avvisiamo nei log

        # --- FASE 2: Chunking (Spezzettamento in paragrafi piccoli per non confondere l'IA) ---
        if all_documents:
            logger.info("Spezzettamento dei documenti in corso...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_documents)

            # --- FASE 3: Embeddings e Vector Store (Ora usa Google per risparmiare RAM!) ---
            logger.info("Creazione del Database Vettoriale (FAISS)...")
            # Usa i server di Google per calcolare gli embeddings, non la memoria di Render!
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        else:
            retriever = None

        # --- FASE 4: Configurazione LLM e Prompt ---
        logger.info("Configurazione del modello principale Gemini...")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        system_prompt = (
            "Sei l'assistente virtuale altamente professionale di 'aipermind lab', un servizio "
            "che offre Validazione Strategica tramite AI Native (impossibile prima del 2023). "
            "Il tuo obiettivo è rispondere alle domande degli utenti in modo chiaro, persuasivo "
            "e basandoti ESCLUSIVAMENTE sul contesto fornito qui sotto.\n"
            "Se l'informazione non è nel contesto, non inventare: di' gentilmente che "
            "l'utente può compilare il modulo per ricevere dettagli specifici via email.\n\n"
            "Contesto aziendale estratto dai PDF e dal Web:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # --- FASE 5: Creazione della Catena (Chain) ---
        if retriever:
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            logger.info("✅ SUCCESS! RAG Inizializzato con successo. Il server è pronto a rispondere.")
        else:
            logger.info("⚠️ Server avviato SENZA documenti di contesto (RAG disabilitato temporaneamente).")
        
    except Exception as e:
        logger.error(f"❌ Errore critico durante l'inizializzazione del RAG: {e}")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="Il sistema si sta caricando o non ci sono PDF validi. Riprova tra poco.")
        
    try:
        # Passa la domanda dell'utente alla catena RAG
        response = rag_chain.invoke({"input": request.message})
        return {"reply": response["answer"]}
    except Exception as e:
        logger.error(f"Errore nella generazione della risposta: {e}")
        raise HTTPException(status_code=500, detail="Errore interno durante l'elaborazione della risposta.")

@app.get("/")
def health_check():
    return {"status": "Backend aipermind RAG operativo ed esposto correttamente sulla rete."}
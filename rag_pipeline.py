import os
import yaml
import logging
import time
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

from llm_loader import get_llm
from loaders.load_all_documents import load_all_documents

# === Init ===
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🧹 Suppress Metal spam logs
class MetalSpamFilter(logging.Filter):
    def filter(self, record):
        return "ggml_metal" not in record.getMessage()

logging.getLogger().addFilter(MetalSpamFilter())

DATA_DIR = os.getenv("DATA_DIR", "data")
TECHNICAL_INFO = os.getenv("TECHNICAL_INFO", "false").lower() == "true"

QA_PROMPT = PromptTemplate.from_template(
"""
You are a support assistant answering client questions using internal documentation and instructions.

Respond only based on the context below. If the answer is not clearly found in the context, say:
"I'm not sure. There is not enough data."

Context may include internal guidance in note format. Do not invent details, and do not repeat the question.

Context:
{context}

Answer:
"""
)

# === Load config YAML ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Chunking & Indexing ===
def build_vectorstore(docs_pdf, docs_sheet, docs_guide, embedding_model):
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks_pdf = splitter.split_documents(docs_pdf)
    chunks_guide = splitter.split_documents(docs_guide)
    chunks = [*docs_sheet, *chunks_guide, *chunks_pdf]

    if not chunks:
        raise RuntimeError("❌ No documents found. Please check config.yaml and ensure data is available.")

    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.from_documents(chunks, embedding)

# === Chain Construction ===
def build_qa_chain(llm, vectorstore, tech_info: bool):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr",  # diversity-aware, reduces duplicate info
            search_kwargs={"k": 2}
        ),
        return_source_documents=tech_info,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

# === Main Builder ===
def build_qa():
    docs_pdf, docs_sheet, docs_guide = load_all_documents(config)

    if not (docs_pdf or docs_sheet or docs_guide):
        raise RuntimeError("❌ No documents found. Check config.yaml.")

    embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    vectorstore = build_vectorstore(docs_pdf, docs_sheet, docs_guide, embedding_model)

    llm = get_llm()
    return build_qa_chain(llm, vectorstore, tech_info=TECHNICAL_INFO)

# === Initialize ===
qa_chain = build_qa()
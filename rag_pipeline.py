import os
import yaml
import logging
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm_loader import get_llm, get_model_config
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

# === Load YAML Config ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = config.get("data_dir", "data")
TECHNICAL_INFO = str(config.get("technical_info", False)).lower() == "true"

# === Prompt ===
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

# === Chunking & Indexing ===
def build_vectorstore(docs_pdf, docs_sheet, docs_guide, embedding_model_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks_pdf = splitter.split_documents(docs_pdf)
    chunks_guide = splitter.split_documents(docs_guide)
    chunks = [*docs_sheet, *chunks_guide, *chunks_pdf]

    if not chunks:
        raise RuntimeError("❌ No documents found. Please check config.yaml and ensure data is available.")

    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return FAISS.from_documents(chunks, embedding)

# === Chain Construction ===
def build_qa_chain(llm, vectorstore, tech_info: bool):
    model_config = get_model_config()
    retrieval_cfg = model_config.get("retrieval_strategy", {})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type=retrieval_cfg.get("search_type", "mmr"),
            search_kwargs={"k": retrieval_cfg.get("search_k", 2)}
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

    embedding_profiles = config.get("embedding_profiles", {})
    active_embedding = config.get("embedding_profile", "bge-small")

    if active_embedding not in embedding_profiles:
        raise ValueError(f"❌ Embedding profile '{active_embedding}' not found in config.yaml")

    embedding_model_name = embedding_profiles[active_embedding]["model_name"]
    vectorstore = build_vectorstore(docs_pdf, docs_sheet, docs_guide, embedding_model_name)

    llm = get_llm()
    return build_qa_chain(llm, vectorstore, tech_info=TECHNICAL_INFO)

# === Initialize ===
qa_chain = build_qa()
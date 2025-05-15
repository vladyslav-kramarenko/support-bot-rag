import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

DATA_DIR = os.getenv("DATA_DIR", "data")

load_dotenv()

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Load PDFs
from manual_file_loader import download_file_from_drive
docs_pdf = []
for item in config.get("pdfs", []):
    try:
        download_file_from_drive(item["file_id"], item["filename"])
        path = os.path.join(DATA_DIR, item["filename"])
        loader = PyPDFLoader(path)
        docs_pdf.extend(loader.load())
    except Exception as e:
        print(f"❌ Failed to load PDF {item.get('filename')}: {e}")



# Load Google Sheets
from google_sheet_loader import load_google_sheet
docs_sheet = []
for item in config.get("sheets", []):
    try:
        docs_sheet.extend(load_google_sheet(item["url"]))
    except Exception as e:
        print(f"❌ Failed to load Sheet: {item.get('url')}\n{e}")



# Load Google Docs
from download_doc import download_google_doc
from langchain_community.document_loaders import TextLoader
docs_guide = []
for item in config.get("docs", []):
    try:
        path = download_google_doc(item["file_id"], item["filename"])
        loader = TextLoader(path)
        docs_guide.extend(loader.load())
    except Exception as e:
        print(f"❌ Failed to load Doc {item.get('filename')}: {e}")

if not (docs_pdf or docs_sheet or docs_guide):
    raise RuntimeError("❌ No documents found. Please check config.yaml for 'pdfs', 'sheets', or 'docs'.")

# === Split only long-form documents (PDF and Google Doc) ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks_pdf = splitter.split_documents(docs_pdf)
chunks_guide = splitter.split_documents(docs_guide)

# === Do NOT split structured table content (Google Sheet) ===
chunks_sheet = docs_sheet

# === Merge all documents together ===
# chunks = chunks_pdf + chunks_guide + chunks_sheet
chunks = chunks_sheet + chunks_guide

# === Use local embeddings ===
embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
embedding = HuggingFaceEmbeddings(model_name=embedding_model)

# === Create vector index ===
db = FAISS.from_documents(chunks, embedding)

# === Load local LLM for generation ===
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(\
    model_path=os.path.abspath(os.getenv("LLM_MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")),
    temperature=0.1,
    max_tokens=512,
    n_ctx=4096,
    verbose=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False
)
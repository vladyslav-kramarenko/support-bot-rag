import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


# === Load environment ===
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "data")
TECHNICAL_INFO = os.getenv("TECHNICAL_INFO", "false").lower() == "true"

# === Load config YAML ===
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# === Load PDFs ===
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


# === Load Google Sheets ===
from google_sheet_loader import load_google_sheet

docs_sheet = []
for item in config.get("sheets", []):
    try:
        docs_sheet.extend(load_google_sheet(item["url"]))
    except Exception as e:
        print(f"❌ Failed to load Sheet: {item.get('url')}\n{e}")


# === Load Google Docs ===
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

# === Split documents ===
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks_pdf = splitter.split_documents(docs_pdf)
chunks_guide = splitter.split_documents(docs_guide)
chunks_sheet = docs_sheet  # no splitting for structured data
chunks = chunks_sheet + chunks_guide + chunks_pdf

# === Embeddings ===
embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
embedding = HuggingFaceEmbeddings(model_name=embedding_model)
db = FAISS.from_documents(chunks, embedding)

# === Load LLM ===
llm = LlamaCpp(
    model_path=os.path.abspath(os.getenv("LLM_MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")),  # Path to the local GGUF model file
    temperature=0.1,              # Controls randomness: lower = more deterministic, higher = more creative. 0.1 gives reliable, focused answers
    max_tokens=512,               # Maximum number of tokens the model is allowed to generate in a response
    n_ctx=2048,                   # Size of the context window (max prompt + output tokens). Should match model capabilities
    n_batch=64,                   # Number of tokens to evaluate in parallel (helps speed). Tune based on available RAM/VRAM
    n_threads=8,                  # Number of CPU threads to use for inference. Match to your physical CPU cores (M3 = 8 performance cores)
    repeat_penalty=1.15,          # Penalizes repetition. Values >1.0 discourage repeating tokens. Default is usually 1.1
    repeat_last_n=64,             # How many of the last tokens to apply repeat_penalty to. Higher helps prevent long repetitions
    top_k=40,                     # Restricts token sampling to top_k most likely tokens. Lower = more focused/less diverse output
    top_p=0.9,                    # Nucleus sampling: includes tokens with cumulative probability up to top_p. Controls diversity
    verbose=True                  # Print model loading and inference details (helpful for debugging and performance monitoring)
)

# === Custom QA Chain with Prompt ===
custom_prompt = PromptTemplate.from_template(
    """
    You are a helpful support assistant for a siding company.
    Use the following context to answer the question.
    If the answer is unknown or not clearly stated, say you don't know.

    Context:
    {context}

    Question: {question}
    Helpful Answer:
    """
)

combine_chain = load_qa_with_sources_chain(
    llm,
    prompt=custom_prompt,
    document_variable_name="context"  # matches your prompt input
)

qa_chain = RetrievalQA(
    combine_documents_chain=combine_chain,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=TECHNICAL_INFO,
)
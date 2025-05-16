import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from loaders.pdf_loader import download_file_from_drive
from loaders.gsheet_loader import load_google_sheet
from loaders.gdoc_loader import download_google_doc

logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "data")

def load_pdfs(pdfs_config: list) -> list:
    """Load PDF documents from Google Drive."""
    docs = []
    for item in pdfs_config:
        try:
            download_file_from_drive(item["file_id"], item["filename"])
            path = os.path.join(DATA_DIR, item["filename"])
            docs.extend(PyPDFLoader(path).load())
        except Exception as e:
            logger.warning("❌ Failed to load PDF %s: %s", item.get("filename"), e)
    return docs

def load_sheets(sheets_config: list) -> list:
    """Load documents from Google Sheets."""
    docs = []
    for item in sheets_config:
        try:
            docs.extend(load_google_sheet(item["url"]))
        except Exception as e:
            logger.warning("❌ Failed to load Sheet %s: %s", item.get("url"), e)
    return docs

def load_docs(docs_config: list) -> list:
    """Load Google Docs as text."""
    docs = []
    for item in docs_config:
        try:
            path = download_google_doc(item["file_id"], item["filename"])
            docs.extend(TextLoader(path).load())
        except Exception as e:
            logger.warning("❌ Failed to load Doc %s: %s", item.get("filename"), e)
    return docs

def load_all_documents(config: dict) -> tuple[list, list, list]:
    """
    Wrapper that loads all document types (PDF, Sheets, Docs) from a config dict.
    Returns: tuple (pdf_docs, sheet_docs, guide_docs)
    """
    pdf_docs = load_pdfs(config.get("pdfs", []))
    sheet_docs = load_sheets(config.get("sheets", []))
    guide_docs = load_docs(config.get("docs", []))
    return pdf_docs, sheet_docs, guide_docs
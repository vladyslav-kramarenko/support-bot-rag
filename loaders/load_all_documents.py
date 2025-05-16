import os
import logging
import yaml
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from loaders.gdrive_file_loader import download_file_from_drive
from loaders.gsheet_loader import load_google_sheet
from loaders.gdoc_loader import download_google_doc
# from loaders.pdf_image_loader import load_pdf_with_images
from loaders.pdf_ocr_loader import load_pdf_with_images

logger = logging.getLogger(__name__)

# === Load YAML Config ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = config.get("data_dir", "data")

def load_pdfs(pdfs_config: list) -> list:
    """Load PDF documents from Google Drive."""
    docs = []
    for item in pdfs_config:
        try:
            download_file_from_drive(item["file_id"], item["filename"])
            path = os.path.join(DATA_DIR, item["filename"])
            docs.extend(PyPDFLoader(path).load())
            # docs.extend(load_pdf_with_images(path))
        except Exception as e:
            logger.warning("❌ Failed to load PDF %s: %s", item.get("filename"), e)
    return docs

# def load_pdfs(pdfs_config: list) -> list:
#     """Load PDF documents with text and image-based OCR content from Google Drive."""
#     docs = []
#     for item in pdfs_config:
#         try:
#             download_file_from_drive(item["file_id"], item["filename"])
#             path = os.path.join(DATA_DIR, item["filename"])
#             docs.extend(load_pdf_with_images(path))  # Use OCR-enabled loader
#         except Exception as e:
#             logger.warning("❌ Failed to load PDF %s: %s", item.get("filename"), e)
#     return docs

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
    data_sources = config.get("data_sources", {})
    pdf_docs = load_pdfs(data_sources.get("pdfs", []))
    sheet_docs = load_sheets(data_sources.get("sheets", []))
    guide_docs = load_docs(data_sources.get("docs", []))
    return pdf_docs, sheet_docs, guide_docs
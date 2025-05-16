import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain.schema import Document
import os

def load_pdf_with_images(pdf_path: str) -> list[Document]:
    """
    Extracts text and OCR from a PDF. Combines page text and OCR into a single Document.
    OCR image blocks are prepended with [Screenshot Text] for LLM awareness.
    """
    docs = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        ocr_blocks = []

        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            try:
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    ocr_blocks.append(f"[Screenshot Text]:\n{ocr_text.strip()}")
            except Exception as e:
                continue  # Skip failed OCR without crashing

        # Combine original text and OCR blocks
        full_content = text
        if ocr_blocks:
            full_content += "\n\n" + "\n\n".join(ocr_blocks)

        if full_content.strip():
            docs.append(Document(
                page_content=full_content[:2000],  # Optional: trim overly long pages
                metadata={"source": pdf_path, "page": page_num + 1}
            ))

    return docs


# Optional: debug runner
if __name__ == "__main__":
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

    pdf_file = "data/ServiceMinder - Manual.pdf"
    documents = load_pdf_with_images(pdf_file)

    print(f"Extracted {len(documents)} documents")
    for doc in documents:
        print(f"--- {doc.metadata} ---")
        print(doc.page_content[:500])
        print("\n")

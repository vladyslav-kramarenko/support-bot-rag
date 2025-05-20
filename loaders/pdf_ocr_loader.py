import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain.schema import Document

def load_pdf_with_images(pdf_path: str) -> list[Document]:
    docs = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        image_texts = []
        images = page.get_images(full=True)

        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Run OCR
            ocr_text = pytesseract.image_to_string(image)
            if ocr_text.strip():
                image_texts.append(f"[Image {img_index + 1} OCR]:\n{ocr_text.strip()}")

        full_text = text.strip()
        if image_texts:
            full_text += "\n\n" + "\n\n".join(image_texts)

        if full_text:
            docs.append(Document(page_content=full_text, metadata={"source": pdf_path, "page": page_num + 1}))

    return docs


# Example usage
if __name__ == "__main__":
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

    pdf_file = "data/sample_manual_with_images.pdf"
    documents = load_pdf_with_images(pdf_file)

    for doc in documents[:2]:  # Show first two
        print(f"--- Page {doc.metadata['page']} ---")
        print(doc.page_content[:500])  # First 500 chars

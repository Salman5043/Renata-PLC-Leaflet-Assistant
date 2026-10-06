from pathlib import Path
import re
import shutil
import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DOCS_DIR
from .embeddings import get_embeddings

SECTION_PATTERN = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")

def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("￾", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def detect_product(lines: list[str], filename: str) -> tuple[str, str]:
    product = Path(filename).stem
    ingredient = "Unknown"
    for i, line in enumerate(lines[:50]):
        low = line.lower().strip(" :-")
        if low == "brand name" and i + 1 < len(lines):
            product = lines[i + 1].strip()
        elif low == "active ingredient" and i + 1 < len(lines):
            ingredient = lines[i + 1].strip()
        elif low.startswith("brand name"):
            value = line.split("Brand name", 1)[-1].strip(" :-")
            if value:
                product = value
        elif low.startswith("active ingredient"):
            value = line.split("Active ingredient", 1)[-1].strip(" :-")
            if value:
                ingredient = value
    return product, ingredient

def extract_documents(pdf_path: Path) -> list[Document]:
    pdf = pymupdf.open(pdf_path)
    try:
        first = clean_text(pdf[0].get_text("text")) if len(pdf) else ""
        product, ingredient = detect_product([x.strip() for x in first.splitlines() if x.strip()], pdf_path.name)
        docs: list[Document] = []
        current_section = "General"
        for page_number, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if not text:
                continue
            lines = [x.strip() for x in text.splitlines() if x.strip()]
            buffer: list[str] = []
            def flush() -> None:
                if not buffer:
                    return
                content = "\n".join(buffer).strip()
                if not content:
                    return
                docs.append(Document(
                    page_content=(
                        f"Product: {product}\n"
                        f"Active ingredient: {ingredient}\n"
                        f"Section: {current_section}\n"
                        f"Page: {page_number}\n\n{content}"
                    ),
                    metadata={
                        "source": pdf_path.name,
                        "product": product,
                        "active_ingredient": ingredient,
                        "page": page_number,
                        "section": current_section,
                    },
                ))
            for line in lines:
                match = SECTION_PATTERN.match(line)
                if match:
                    flush(); buffer.clear(); current_section = match.group(2).strip()
                else:
                    buffer.append(line)
            flush()
        return docs
    finally:
        pdf.close()

def load_pdfs() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(DOCS_DIR.glob("*.pdf")):
        docs.extend(extract_documents(path))
    return docs

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks

def build_vectorstore() -> Chroma:
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}")
    documents = load_pdfs()
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("No text could be extracted from the PDFs")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(chunks)
    print(f"Indexed PDFs: {len(pdfs)}")
    print(f"Section/page blocks: {len(documents)}")
    print(f"Vector chunks: {len(chunks)}")
    return store

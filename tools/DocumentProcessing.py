from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pdf_loader(path):
    loader = PyPDFLoader(file_path=path)
    pdf_document = loader.load()
    # Debugging output
    # print(f"✅ Loaded {len(pdf_document)} pages from PDF")
    # for i, page in enumerate(pdf_document):
    #     print(f"Page {i+1} content: {page.page_content[:200]}...")
    return pdf_document


def directory_loader(path):
    loader = DirectoryLoader(path=path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs


def document_split(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(doc)
    print(f"✅ Split into {len(texts)} chunks")
    for i, chunk in enumerate(texts[:5]):  # Show first 5 chunks
        print(f"Chunk {i + 1}: {chunk.page_content[:200]}...")
    return texts

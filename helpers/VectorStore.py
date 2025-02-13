from langchain_chroma import Chroma

from helpers.Models import get_embedding
from tools.DocumentProcessing import document_split

persist_directory = 'files/chroma/'


def create_vector_db(doc, embedding_model):
    vectordb = Chroma.from_documents(
        documents=doc,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    return vectordb



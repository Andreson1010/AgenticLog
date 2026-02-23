# AgenticLog - Pipeline RAG
"""
Cria banco de dados vetorial a partir de documentos JSON.
Execute: python -m agenticlog.rag
"""

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch

from agenticlog.config import (
    DIR_DOCUMENTS,
    DIR_VECTORDB,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

vectordb = None


def cria_vectordb():
    """Cria o banco vetorial (ChromaDB) a partir dos documentos em data/documents/."""
    global vectordb

    print("\nGerando as Embeddings. Aguarde...")

    jq_schema = 'to_entries | map(.key + ": " + .value) | join("\\n")'
    loader = DirectoryLoader(
        str(DIR_DOCUMENTS),
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": jq_schema},
    )
    documents = loader.load()

    if not documents:
        print("Nenhum documento encontrado.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    global vectordb
    vectordb = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=str(DIR_VECTORDB),
    )

    print("\nBanco de Dados Vetorial do RAG Criado com Sucesso.\n")


if __name__ == "__main__":
    cria_vectordb()

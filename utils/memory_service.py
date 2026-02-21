from __future__ import annotations
import chromadb
from chromadb.config import Settings
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import pathlib


class MemoryService:
    def __init__(self, persist_directory: str = "~/.codex/memory_db"):
        resolved = pathlib.Path(persist_directory).expanduser()
        self.persist_directory = str(resolved)
        
        self.client = chromadb.PersistentClient(path=str(resolved))
        self.collection = self.client.get_or_create_collection("memories")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
    
    def store(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        chunks = self.text_splitter.split_text(text)
        chunk_ids = [f"chunk_{i}_{hash(chunk) % 100000}" for i, chunk in enumerate(chunks)]
        embeddings = self.embeddings.embed_documents(chunks)
        
        metadatas = [metadata or {} for _ in chunks]
        
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "text": doc,
                    "distance": results["distances"][0][i] if "distances" in results else None,
                    "metadata": results["metadatas"][0][i] if "metadatas" in results else {}
                })
        
        return output
    
    def stats(self) -> dict[str, Any]:
        return {
            "count": self.collection.count(),
            "persist_directory": self.persist_directory
        }
    
    def clear(self) -> None:
        self.client.delete_collection("memories")
        self.collection = self.client.get_or_create_collection("memories")
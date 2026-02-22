from __future__ import annotations
import chromadb
from chromadb.config import Settings
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
        if not text or not text.strip():
            raise ValueError("text cannot be empty")
        
        try:
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
        except Exception as e:
            raise RuntimeError(f"Failed to store text: {e}") from e
    
    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve: {e}") from e
        
        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "text": doc,
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}
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

    def delete_by_metadata(self, filter: dict[str, Any]) -> None:
        """Delete entries that match the metadata filter."""
        try:
            self.collection.delete(where=filter)
        except Exception as e:
            raise RuntimeError(f"Failed to delete by metadata: {e}") from e

    def get_by_metadata(self, filter: dict[str, Any]) -> list[dict[str, Any]]:
        """Get entries that match the metadata filter."""
        try:
            results = self.collection.get(where=filter)
            output = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    output.append({
                        "id": results["ids"][i],
                        "text": doc,
                        "metadata": results["metadatas"][i] if results.get("metadatas") else {}
                    })
            return output
        except Exception as e:
            raise RuntimeError(f"Failed to get by metadata: {e}") from e

import pathlib
from typing import List, Dict, Any, Optional
from openmemory.client import Memory
from openmemory.core.config import env
from openmemory.core.db import db

class CognitiveMemory:
    def __init__(self, db_path: str = "~/.codex/cognitive_memory.db"):
        resolved = pathlib.Path(db_path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        
        db_url = f"sqlite:///{resolved}"
        env.database_url = db_url
        
        # Reset connection to use the new URL
        db.conn = None
        db.connect()
        
        # Ensure the file is created by the DB connection
        resolved.touch(exist_ok=True)
        
        self.client = Memory()

    async def add(self, text: str, user_id: str = "default", doc_type: str = "fact", tags: Optional[List[str]] = None):
        if not text or not text.strip():
            raise ValueError("text cannot be empty")
        
        # Structure metadata as required by spec
        meta = {"type": doc_type}
        
        # SDK uses 'content' parameter for text, passes tags/meta as kwargs
        return await self.client.add(content=text, user_id=user_id, meta=meta, tags=tags)

    async def search(self, query: str, user_id: str = "default", k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        return await self.client.search(query, user_id=user_id, limit=k)

    async def delete(self, memory_id: str):
        if not memory_id:
            raise ValueError("memory_id cannot be empty")
        return await self.client.delete(memory_id)

    async def stats(self) -> Dict[str, Any]:
        return {
            "engine": "OpenMemory",
            "storage": "SQLite",
            "database_url": env.database_url
        }

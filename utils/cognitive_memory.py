import asyncio
import pathlib
from typing import List, Dict, Any, Optional
from openmemory.client import Memory
from openmemory.core.config import env
from openmemory.core.db import db

class CognitiveMemory:
    def __init__(self, db_path: str = "~/.codex/cognitive_memory.db"):
        resolved = pathlib.Path(db_path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        
        # OpenMemory uses a singleton for DB connection.
        # We need to set the database URL before any connection is made,
        # or reset the connection if it's already established.
        env.database_url = f"sqlite:///{resolved}"
        if db.conn:
            db.conn.close()
            db.conn = None
        db.connect()
        
        self.client = Memory()

    async def add(self, content: str, user_id: str = "default", type: str = "fact", tags: List[str] = None):
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        # Memory.add uses **kwargs to pass meta and tags to ingest_document
        await self.client.add(content, user_id=user_id, meta={"type": type}, tags=tags or [])

    async def search(self, query: str, user_id: str = "default", k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        return await self.client.search(query, user_id=user_id, limit=k)

    async def stats(self) -> Dict[str, Any]:
        return {"engine": "OpenMemory", "storage": "SQLite"}

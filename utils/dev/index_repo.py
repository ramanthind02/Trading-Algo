import os
import sys
import pathlib
import time
from typing import List

# Add project root to sys.path
_FILE_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in _FILE_PATH.parents if (parent / ".git").exists()),
    _FILE_PATH.parents[2],
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.memory.memory_service import MemoryService

class RepoIndexer:
    def __init__(self, repo_path: str = "."):
        self.repo_path = pathlib.Path(repo_path).absolute()
        self.service = MemoryService()
        self.supported_extensions = {'.py', '.md', '.txt', '.json', '.toml', '.yaml', '.yml'}
        self.ignored_dirs = {'venv', '.git', '.worktrees', '__pycache__', 'node_modules', '.pytest_cache'}

    def should_index(self, path: pathlib.Path) -> bool:
        if path.suffix not in self.supported_extensions:
            return False
        for ignored in self.ignored_dirs:
            if ignored in path.parts:
                return False
        return True

    def get_files_to_index(self) -> List[pathlib.Path]:
        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
            
            for filename in filenames:
                path = pathlib.Path(root) / filename
                if self.should_index(path):
                    files.append(path)
        return files

    def index_repo(self, incremental: bool = True):
        files = self.get_files_to_index()
        print(f"Found {len(files)} files to index.")
        
        for i, file_path in enumerate(files, 1):
            rel_path = file_path.relative_to(self.repo_path)
            last_modified = file_path.stat().st_mtime
            
            if incremental:
                # Check if file has changed in DB
                try:
                    existing = self.service.get_by_metadata({"file_path": str(rel_path)})
                    if existing:
                        db_last_modified = existing[0]["metadata"].get("last_modified", 0)
                        if last_modified <= db_last_modified:
                            print(f"[{i}/{len(files)}] Skipping {rel_path} (already up to date).")
                            continue
                        else:
                            print(f"[{i}/{len(files)}] Updating {rel_path} (file changed).")
                            self.service.delete_by_metadata({"file_path": str(rel_path)})
                    else:
                        print(f"[{i}/{len(files)}] Indexing new file {rel_path}...")
                except Exception as e:
                    print(f"Error checking DB for {rel_path}: {e}")
                    # If check fails, we still try to index, but maybe we should skip.
                    # For now, let's proceed.
            else:
                print(f"[{i}/{len(files)}] Re-indexing {rel_path}...")
                try:
                    self.service.delete_by_metadata({"file_path": str(rel_path)})
                except Exception:
                    pass

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Metadata for the file
                metadata = {
                    "file_path": str(rel_path),
                    "last_modified": last_modified,
                    "source": "repo_indexer"
                }
                
                self.service.store(content, metadata=metadata)
                
            except Exception as e:
                print(f"Error indexing {rel_path}: {e}")

if __name__ == "__main__":
    indexer = RepoIndexer()
    indexer.index_repo()

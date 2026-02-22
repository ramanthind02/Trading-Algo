import pytest
import pathlib
from utils.index_repo import RepoIndexer
from unittest.mock import MagicMock, patch

def test_indexer_should_index():
    indexer = RepoIndexer()
    assert indexer.should_index(pathlib.Path("test.py")) is True
    assert indexer.should_index(pathlib.Path(".git/config")) is False
    assert indexer.should_index(pathlib.Path("venv/bin/python")) is False
    assert indexer.should_index(pathlib.Path("sub/dir/.worktrees/rag/file.py")) is False

@patch('utils.index_repo.MemoryService')
def test_indexer_run(mock_service_class, tmp_path):
    # Create dummy file
    test_file = tmp_path / "test_file.py"
    test_file.write_text("print('hello world')")
    
    indexer = RepoIndexer(repo_path=str(tmp_path))
    indexer.index_repo()
    
    # Verify MemoryService was used
    instance = mock_service_class.return_value
    assert instance.store.called
    
    # Check that it was called with content and metadata
    args, kwargs = instance.store.call_args
    assert args[0] == "print('hello world')"
    assert kwargs['metadata']['file_path'] == "test_file.py"
    assert 'last_modified' in kwargs['metadata']

@patch('utils.index_repo.MemoryService')
def test_incremental_indexing_skips_unchanged(mock_service_class, tmp_path):
    test_file = tmp_path / "test_file.py"
    test_file.write_text("print('hello world')")
    last_mod = test_file.stat().st_mtime
    
    instance = mock_service_class.return_value
    # Mock existing entry with same last_modified
    instance.get_by_metadata.return_value = [{"metadata": {"last_modified": last_mod}}]
    
    indexer = RepoIndexer(repo_path=str(tmp_path))
    indexer.index_repo(incremental=True)
    
    # Store should NOT be called because file is up to date
    assert not instance.store.called

@patch('utils.index_repo.MemoryService')
def test_incremental_indexing_updates_changed(mock_service_class, tmp_path):
    test_file = tmp_path / "test_file.py"
    test_file.write_text("print('hello world')")
    last_mod = test_file.stat().st_mtime
    
    instance = mock_service_class.return_value
    # Mock existing entry with OLDER last_modified
    instance.get_by_metadata.return_value = [{"metadata": {"last_modified": last_mod - 10}}]
    
    indexer = RepoIndexer(repo_path=str(tmp_path))
    indexer.index_repo(incremental=True)
    
    # Delete and then Store should be called
    assert instance.delete_by_metadata.called
    assert instance.store.called

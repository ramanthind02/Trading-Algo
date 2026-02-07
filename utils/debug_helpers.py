"""
Debug helpers for safe JSON serialization in debug logging.

This module provides utilities to safely serialize various Python objects
(enums, numpy types, pandas types, etc.) to JSON-serializable formats.
"""

import json
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd


def safe_json_value(obj: Any) -> Any:
    """
    Convert any object to a JSON-serializable value.
    
    Handles common non-serializable types:
    - Enum: converts to .name
    - numpy types: converts to native Python types
    - pandas types: converts to native Python types
    - datetime: converts to ISO string
    - None/NaN/inf: preserves or converts to null
    
    Parameters
    ----------
    obj : Any
        Object to convert
        
    Returns
    -------
    Any
        JSON-serializable version of the object
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle Enum types (including Ticker)
    if isinstance(obj, Enum):
        return obj.name
    
    # Handle numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        # Handle NaN and inf
        if np.isnan(val):
            return None
        if np.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle pandas types
    if isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
        return str(obj)
    
    # Handle pandas Series (convert to list)
    if isinstance(obj, pd.Series):
        return obj.tolist()
    
    # Check for pandas NA (only for scalar values)
    try:
        if pd.api.types.is_scalar(obj) and pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass  # Not a scalar, continue
    
    # Handle datetime
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    
    # Handle dict/list recursively
    if isinstance(obj, dict):
        # Convert dict keys to strings if they're not JSON-serializable
        return {str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k: safe_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json_value(item) for item in obj]
    
    # Handle float NaN/inf
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    
    # Return as-is if already serializable (str, int, float, bool)
    return obj


def safe_json_dumps(data: dict) -> str:
    """
    Safely serialize a dict to JSON string, handling non-serializable types.
    
    Parameters
    ----------
    data : dict
        Dictionary to serialize
        
    Returns
    -------
    str
        JSON string
    """
    # Convert all values to safe types
    safe_data = safe_json_value(data)
    return json.dumps(safe_data)


def write_debug_log(filepath: str, location: str, message: str, data: dict, 
                    session_id: str = 'debug-session', run_id: str = 'initial', 
                    hypothesis_id: str = None) -> None:
    """
    Write a debug log entry to file, safely handling all types.
    
    Parameters
    ----------
    filepath : str
        Path to log file
    location : str
        Code location (file:line)
    message : str
        Log message
    data : dict
        Data to log (will be safely converted)
    session_id : str, optional
        Session ID
    run_id : str, optional
        Run ID
    hypothesis_id : str, optional
        Hypothesis ID
    """
    import time
    
    log_entry = {
        'location': location,
        'message': message,
        'data': data,
        'timestamp': int(time.time() * 1000),
        'sessionId': session_id,
        'runId': run_id
    }
    
    if hypothesis_id:
        log_entry['hypothesisId'] = hypothesis_id
    
    try:
        with open(filepath, 'a') as f:
            f.write(safe_json_dumps(log_entry) + '\n')
    except Exception as e:
        # Fallback: write error to log
        try:
            with open(filepath, 'a') as f:
                error_entry = {
                    'location': location,
                    'message': 'Debug log serialization error',
                    'data': {'error': str(e), 'original_message': message},
                    'timestamp': int(time.time() * 1000),
                    'sessionId': session_id,
                    'runId': run_id
                }
                f.write(json.dumps(error_entry) + '\n')
        except:
            pass  # Give up silently

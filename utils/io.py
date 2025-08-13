"""
I/O helpers to save JSON atomically.
"""

import json
from pathlib import Path
import tempfile
import shutil
import logging
from typing import Any, Union

logger = logging.getLogger(__name__)


def save_json_atomic(obj: Any, dest: Union[str, Path]) -> None:
    """
    Atomically save an object as JSON to a destination file.
    
    Args:
        obj: The object to serialize to JSON
        dest: Destination file path as string or Path
        
    Raises:
        OSError: If file operations fail
        TypeError: If object is not JSON serializable
    """
    try:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(dest.parent), suffix=".tmp") as tf:
            try:
                json.dump(obj, tf, indent=2)
                tmp = tf.name
            except TypeError as e:
                raise TypeError(f"Object is not JSON serializable: {e}")
                
        shutil.move(tmp, str(dest))
        logger.info("Saved JSON to %s", dest)
        
    except OSError as e:
        logger.error("Failed to save JSON to %s: %s", dest, e)
        raise
"""core/math_utils.py
--------------------
Shared mathematical utilities used across agents and graph nodes.
Centralises implementations to avoid duplication.
"""

import numpy as np


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns 0.0 for zero-norm vectors to avoid division by zero.
    """
    if not v1 or not v2:
        return 0.0
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0

def dot_product(v1: list[float], v2: list[float]) -> float:
    """
    Compute dot product between two vectors.
    """
    if not v1 or not v2:
        return 0.0
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    return float(np.dot(a, b))

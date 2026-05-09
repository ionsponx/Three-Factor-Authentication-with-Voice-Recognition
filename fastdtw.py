"""Small local fallback for the fastdtw package.

The original project imported the third-party fastdtw package. That package often
needs native compilation on Windows, so this project keeps a pure-Python DTW
implementation with the same return shape used by main.py: (distance, path).
"""

import os

import numpy as np


def _distance(left, right, dist):
    if dist is None or getattr(dist, "__name__", "") == "euclidean":
        diff = left - right
        return float(np.sqrt(np.dot(diff, diff)))
    return float(dist(left, right))


def fastdtw(x, y, radius=1, dist=None):
    """Return DTW distance and a lightweight path placeholder.

    main.py only uses len(path), so the returned path intentionally stores empty
    placeholders instead of every coordinate pair.
    """
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)

    if left.ndim == 1:
        left = left.reshape(-1, 1)
    if right.ndim == 1:
        right = right.reshape(-1, 1)

    n = len(left)
    m = len(right)
    if n == 0 or m == 0:
        return float("inf"), []

    window = int(os.getenv("DTW_WINDOW_FRAMES", "200"))
    window = max(window, radius, abs(n - m) + 2)
    full_matrix = window >= max(n, m)

    previous = {0: (0.0, 0)}
    for i in range(1, n + 1):
        current = {}
        if full_matrix:
            j_start, j_end = 1, m
        else:
            center = round(i * m / n)
            j_start = max(1, center - window)
            j_end = min(m, center + window)

        for j in range(j_start, j_end + 1):
            candidates = [
                previous.get(j),
                current.get(j - 1),
                previous.get(j - 1),
            ]
            candidates = [candidate for candidate in candidates if candidate is not None]
            if not candidates:
                continue

            best_cost, best_steps = min(candidates, key=lambda item: item[0])
            current[j] = (
                best_cost + _distance(left[i - 1], right[j - 1], dist),
                best_steps + 1,
            )
        previous = current

    total_cost, path_length = previous.get(m, (float("inf"), 0))
    return total_cost, [None] * path_length

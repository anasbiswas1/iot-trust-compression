
"""Small deployment measurement helpers used after the algorithmic gates."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import statistics
import time

import numpy as np
import torch
from torch import nn


def serialized_state_dict_bytes(model: nn.Module) -> int:
    with NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save(model.state_dict(), path)
        return int(path.stat().st_size)
    finally:
        path.unlink(missing_ok=True)


def benchmark_latency(
    model: nn.Module,
    example_input: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    warmup: int = 50,
    repeats: int = 200,
    num_threads: int | None = 1,
) -> dict[str, float]:
    device = torch.device(device)
    if device.type == "cpu" and num_threads is not None:
        torch.set_num_threads(int(num_threads))
    model = model.to(device).eval()
    x = example_input.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter_ns() - start) / 1e6)
    arr = np.asarray(timings, dtype=float)
    return {
        "batch_size": int(x.shape[0]),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "throughput_items_per_s": float(x.shape[0] / (np.median(arr) / 1000.0)),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "device": str(device),
        "num_threads": int(torch.get_num_threads()),
    }

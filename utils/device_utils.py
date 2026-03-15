#
# Vendor-agnostic device utilities for FastGS.
# Supports NVIDIA CUDA, AMD ROCm (via PyTorch HIP), Apple MPS, and CPU.
#

import os
import torch


def get_device() -> torch.device:
    """Return the best available compute device.

    Resolution order:
    1. ``FASTGS_DEVICE`` environment variable (e.g. ``cuda``, ``cuda:1``,
       ``rocm``, ``mps``, ``cpu``).
    2. CUDA / ROCm – both are exposed via ``torch.cuda`` when the ROCm
       build of PyTorch is installed.
    3. Apple Metal Performance Shaders (MPS).
    4. CPU as the universal fallback.
    """
    env = os.environ.get("FASTGS_DEVICE", "").strip()
    if env:
        # Allow ``rocm`` as an alias for ``cuda`` (ROCm exposes the same API)
        if env.lower() == "rocm":
            env = "cuda"
        return torch.device(env)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Module-level singleton so every import gets the same object.
DEVICE: torch.device = get_device()


def synchronize(device: torch.device | None = None) -> None:
    """Synchronize the given (or global) device if applicable."""
    d = device or DEVICE
    if d.type == "cuda":
        torch.cuda.synchronize(d)
    # MPS and CPU are synchronous by default.


def empty_cache(device: torch.device | None = None) -> None:
    """Free unused cached memory on the given (or global) device."""
    d = device or DEVICE
    if d.type == "cuda":
        torch.cuda.empty_cache()
    elif d.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


class DeviceTimer:
    """Lightweight cross-device timer.

    On CUDA/ROCm it wraps ``torch.cuda.Event`` for accurate GPU timing.
    On all other backends it falls back to ``time.perf_counter``.
    """

    def __init__(self, device: torch.device | None = None):
        self._device = device or DEVICE
        self._use_cuda_events = self._device.type == "cuda"
        self._start_event = None
        self._end_event = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    def record_start(self) -> None:
        if self._use_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            import time
            self._start_time = time.perf_counter()

    def record_end(self) -> None:
        if self._use_cuda_events:
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._end_event.record()
        else:
            import time
            self._end_time = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        if self._use_cuda_events:
            if self._start_event is None or self._end_event is None:
                return 0.0
            self._start_event.synchronize()
            self._end_event.synchronize()
            return self._start_event.elapsed_time(self._end_event)
        return (self._end_time - self._start_time) * 1e3

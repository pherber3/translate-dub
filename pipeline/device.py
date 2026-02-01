"""Cross-platform device detection and GPU memory management for PyTorch."""

import gc

import torch


def get_best_device() -> str:
    """Get the best available device for inference.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype_for_device(device: str) -> torch.dtype:
    """Get the best dtype for a given device.

    Args:
        device: Device string ("cuda", "mps", "cpu")

    Returns:
        Best supported dtype for the device
    """
    if device == "cuda":
        return torch.bfloat16
    elif device == "mps":
        # MPS supports float16 but not bfloat16
        return torch.float16
    return torch.float32


def get_device_info() -> dict:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "best_device": get_best_device(),
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def clear_gpu_memory():
    """Free GPU memory between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

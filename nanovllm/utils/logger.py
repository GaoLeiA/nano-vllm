import os
import logging
import time
import torch
from contextlib import contextmanager
from typing import Dict, Any, Optional


# Configure logging level from environment variable
LOG_LEVEL = os.environ.get("NANOVLLM_LOG_LEVEL", "INFO").upper()


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    log_level = level if level else LOG_LEVEL
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    return logger


def log_gpu_memory(logger: logging.Logger, prefix: str = "", device: int = 0):
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    
    try:
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        
        logger.info(
            f"{prefix}GPU Memory [Device {device}]: "
            f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
            f"Max Allocated={max_allocated:.2f}GB"
        )
    except Exception as e:
        logger.debug(f"Failed to get GPU memory info: {e}")


def log_tensor_shapes(logger: logging.Logger, tensors_dict: Dict[str, Any], prefix: str = ""):
    """Log shapes of multiple tensors."""
    shapes_info = []
    for name, tensor in tensors_dict.items():
        if isinstance(tensor, torch.Tensor):
            shapes_info.append(f"{name}={list(tensor.shape)}")
        elif isinstance(tensor, (list, tuple)):
            if len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
                shapes_info.append(f"{name}=[{', '.join([str(list(t.shape)) for t in tensor])}]")
            else:
                shapes_info.append(f"{name}=len({len(tensor)})")
        elif tensor is not None:
            shapes_info.append(f"{name}={tensor}")
    
    if shapes_info:
        logger.info(f"{prefix}{', '.join(shapes_info)}")


@contextmanager
def timed_section(logger: logging.Logger, name: str, log_level: int = logging.INFO):
    """Context manager for timing code sections."""
    start_time = time.perf_counter()
    logger.log(log_level, f"[START] {name}")
    
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        logger.log(log_level, f"[END] {name} - Elapsed: {elapsed:.2f}ms")


def log_communication(logger: logging.Logger, operation: str, tensor_shapes: Dict[str, Any], 
                      rank: int, world_size: int):
    """Log distributed communication operations."""
    shapes_str = ', '.join([f"{k}={list(v.shape) if isinstance(v, torch.Tensor) else v}" 
                           for k, v in tensor_shapes.items()])
    logger.info(
        f"Communication [{operation}] Rank={rank}/{world_size}, {shapes_str}"
    )


def format_memory_mb(bytes_val: int) -> str:
    """Format bytes as MB string."""
    return f"{bytes_val / 1024**2:.2f}MB"


def format_memory_gb(bytes_val: int) -> str:
    """Format bytes as GB string."""
    return f"{bytes_val / 1024**3:.2f}GB"

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_device(preference: str = "auto"):
    """
    Select execution device with graceful CPU fallback.
    """
    if not TORCH_AVAILABLE or torch is None:
        logger.info("PyTorch unavailable in environment. Falling back to CPU mode.")
        return "cpu"
    """
    Select execution device with graceful CPU fallback.
    
    Args:
        preference: Preferred device ('auto', 'cuda', 'cpu', 'mps')
        
    Returns:
        torch.device instance
    """
    pref = str(preference).lower()
    
    if pref == "cuda" and torch.cuda.is_available():
        logger.info("Using requested CUDA GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using requested Apple MPS backend")
        return torch.device("mps")
    elif pref == "cpu":
        logger.info("Using explicit CPU backend")
        return torch.device("cpu")
        
    # Auto selection
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Auto-selected CUDA GPU: {device_name}")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Auto-selected Apple MPS backend")
        return torch.device("mps")
    else:
        logger.info("GPU unavailable. Defaulting to CPU execution.")
        return torch.device("cpu")

from .gpt import generate_with_gpt
from .flan import generate_with_flan
from .gpu import generate_with_gpu

# Registry of available model services
MODEL_SERVICES = {
    "gpt": generate_with_gpt,
    "flan": generate_with_flan,
    "gpu": generate_with_gpu,
}

def get_service(name: str):
    """Fetch a service function by model name."""
    try:
        return MODEL_SERVICES[name]
    except KeyError:
        raise ValueError(f"Invalid model: {name}. Available: {list(MODEL_SERVICES.keys())}")

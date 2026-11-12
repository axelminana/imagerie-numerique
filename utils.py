"""Petites fonctions communes à plusieurs modules (matériel de calcul, images)."""
from __future__ import annotations

import gc
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """Repasse sur le CPU, avec un avertissement, si aucun GPU n'est disponible."""
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            log.warning("Aucun GPU CUDA détecté : calcul sur CPU (plus lent)")
            return "cpu"
    return device


def torch_dtype(device: str):
    """Précision des modèles : float16 sur GPU (moitié moins de mémoire), float32 sur CPU."""
    import torch
    return torch.float16 if device == "cuda" else torch.float32


def dtype_kwargs(device: str) -> dict:
    """Argument de précision pour from_pretrained de transformers.

    transformers 4.56 et plus attend dtype=, les versions précédentes torch_dtype=.
    """
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    return {key: torch_dtype(device)}


def free_gpu_memory() -> None:
    """Libère la mémoire GPU après la suppression d'un modèle (del model)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_rgb(path: Path):
    """Ouvre une image en RGB avec Pillow."""
    from PIL import Image
    with Image.open(path) as image:
        return image.convert("RGB")


def batched(items: list, size: int) -> list[list]:
    """Découpe une liste en lots de taille size (le dernier peut être plus petit)."""
    return [items[i:i + size] for i in range(0, len(items), size)]

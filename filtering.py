"""Étape 2 : filtrage des images avec un modèle vision-langage (responsable : R2).

Garde uniquement les images qui montrent une scène de circulation exploitable
(définition en section 6.2 du plan). Deux méthodes sont comparées : BLIP-2
(demandé par le sujet) et CLIP (alternative plus légère).

Produit filter.csv (colonnes : config.FILTER_COLUMNS).
"""
from __future__ import annotations

import logging
from pathlib import Path

import config

log = logging.getLogger(__name__)


def parse_yes_no(answer: str) -> bool:
    """Règle de décision de BLIP-2 : True si la réponse nettoyée commence par "yes"."""
    return answer.strip().lower().startswith("yes")


# ---------------------------------------------------------------------------
# À implémenter (R2)
# ---------------------------------------------------------------------------
def load_blip2(device: str = "cuda"):
    """Charge le processeur et le modèle BLIP-2 (config.BLIP2_MODEL) en float16 sur GPU.

    Retourne le couple (processor, model).
    """
    raise NotImplementedError("TODO (R2) : chargement de BLIP-2")


def blip2_answer(image_path: Path, prompt: str, processor, model, device: str = "cuda") -> str:
    """Pose la question prompt à BLIP-2 sur une image et retourne la réponse brute."""
    raise NotImplementedError("TODO (R2) : question à BLIP-2")


def load_clip(device: str = "cuda"):
    """Charge le processeur et le modèle CLIP (config.CLIP_MODEL).

    Retourne le couple (processor, model).
    """
    raise NotImplementedError("TODO (R2) : chargement de CLIP")


def clip_score(image_path: Path, processor, model, device: str = "cuda") -> float:
    """Score CLIP d'une image : somme des probabilités des descriptions positives.

    Les descriptions sont config.CLIP_POSITIVE_PROMPTS et config.CLIP_NEGATIVE_PROMPTS.
    """
    raise NotImplementedError("TODO (R2) : score CLIP")


def filter_images(zone_dir: Path, method: str = config.DEFAULT_FILTER,
                  device: str = "cuda") -> Path:
    """Exécute l'étape 2 sur les images sélectionnées de metadata.csv.

    method vaut "blip2", "clip" ou "none" (toutes les images sont gardées).
    Écrit filter.csv dans zone_dir et retourne son chemin.
    """
    import csv

    zone_dir = Path(zone_dir)
    metadata_path = zone_dir / config.METADATA_FILE
    if not metadata_path.exists():
        raise RuntimeError(f"{metadata_path} introuvable : lancez d'abord l'étape download.")
    with open(metadata_path, encoding="utf-8") as handle:
        image_ids = [row["image_id"] for row in csv.DictReader(handle)
                     if row["selected"].strip().lower() == "true" and row["path"]]

    if method == "none":
        rows = [{"image_id": i, "method": "none", "answer": "", "score": "", "keep": True}
                for i in image_ids]
    elif method in ("blip2", "clip"):
        # TODO (R2) : appeler blip2_answer / clip_score pour chaque image
        raise NotImplementedError(
            f"filtre {method!r} pas encore codé (R2) : utilisez --filter none en attendant"
        )
    else:
        raise ValueError(f"méthode inconnue : {method!r} (choix : blip2, clip, none)")

    output = zone_dir / config.FILTER_FILE
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.FILTER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    n_keep = sum(bool(r["keep"]) for r in rows)
    log.info("Filtre %s : %d image(s) gardée(s) sur %d", method, n_keep, len(rows))
    return output

"""Étape 3 (suite) : véhicules stationnés ou en circulation (responsable : R4).

Remplit les colonnes parked et parked_method des détections de véhicules
(section 6.4 du plan). Les piétons ne sont pas concernés.

Stratégies (config.PARKED_STRATEGIES) :
    - "none"     : aucun véhicule n'est marqué stationné
    - "geometry" : règle sur la position et la forme de la boîte
    - "clip"     : question à CLIP sur le recadrage du véhicule
"""
from __future__ import annotations

import logging
from pathlib import Path

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# À implémenter (R4)
# ---------------------------------------------------------------------------
def parked_by_geometry(detection: dict, image_width: int, image_height: int) -> bool:
    """Hypothèse à vérifier : un véhicule garé est souvent sur un bord latéral de
    l'image et vu de profil (boîte nettement plus large que haute).
    """
    raise NotImplementedError("TODO (R4) : règle géométrique")


def parked_by_clip(image_path: Path, detection: dict, processor, model,
                   margin: float = 0.2, device: str = "cuda") -> bool:
    """Découpe la boîte avec une marge et compare "a parked car on the side of the street"
    à "a car driving on the road" avec CLIP.
    """
    raise NotImplementedError("TODO (R4) : question CLIP sur le recadrage")


def mark_parked(detections: list[dict], zone_dir: Path,
                strategy: str = config.DEFAULT_PARKED_STRATEGY,
                device: str = "cuda") -> list[dict]:
    """Remplit parked et parked_method pour les détections de config.VEHICLE_CATEGORIES.

    Retourne la liste de détections complétée.
    """
    raise NotImplementedError("TODO (R4) : tri des véhicules stationnés")

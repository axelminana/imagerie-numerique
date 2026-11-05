"""Étape 3 : détection des usagers avec YOLO (responsable : R3).

Détecte les objets sur les images conservées par le filtre, les regroupe dans
les catégories du projet, marque les personnes qui conduisent un deux-roues
(is_rider), puis appelle parked.py pour marquer les véhicules stationnés.

Produit detections.csv (colonnes : config.DETECTION_COLUMNS).
"""
from __future__ import annotations

import logging
from pathlib import Path

import config

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]   # (x1, y1, x2, y2) en pixels


def coco_to_category(coco_class: str) -> str | None:
    """Catégorie du projet pour une classe COCO, ou None si la classe est ignorée."""
    return config.COCO_TO_CATEGORY.get(coco_class)


def covered_fraction(person_box: Box, vehicle_box: Box) -> float:
    """Part de la boîte de la personne couverte par la boîte du deux-roues (entre 0 et 1)."""
    x1, y1 = max(person_box[0], vehicle_box[0]), max(person_box[1], vehicle_box[1])
    x2, y2 = min(person_box[2], vehicle_box[2]), min(person_box[3], vehicle_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
    return inter / area if area > 0 else 0.0


# ---------------------------------------------------------------------------
# À implémenter (R3)
# ---------------------------------------------------------------------------
def mark_riders(detections: list[dict],
                threshold: float = config.RIDER_OVERLAP_THRESHOLD) -> list[dict]:
    """Met is_rider à True pour les personnes qui conduisent un deux-roues.

    Une personne est conductrice si covered_fraction(personne, deux-roues) dépasse
    threshold et si son centre horizontal tombe dans la boîte du deux-roues.
    Un vélo est associé à une personne au plus, une moto à deux personnes au plus.
    Les détections sont celles d'une seule image (dictionnaires au format detections.csv).
    """
    raise NotImplementedError("TODO (R3) : dédoublonnage des conducteurs")


def run_yolo(image_paths: list[Path], model_path: str, conf: float,
             imgsz: int, device: str) -> list[dict]:
    """Lance YOLO (ultralytics) par lots et retourne une détection par objet retenu.

    Seules les classes de config.COCO_TO_CATEGORY sont gardées.
    """
    raise NotImplementedError("TODO (R3) : inférence YOLO")


def detect(zone_dir: Path, model_path: str = config.DEFAULT_DETECTOR,
           conf: float = config.DEFAULT_CONF, imgsz: int = config.DEFAULT_IMGSZ,
           device: str = "cuda",
           parked_strategy: str = config.DEFAULT_PARKED_STRATEGY) -> Path:
    """Exécute l'étape 3 sur les images gardées dans filter.csv.

    Enchaîne run_yolo, mark_riders, puis parked.mark_parked. Écrit detections.csv
    dans zone_dir et retourne son chemin.
    """
    raise NotImplementedError("TODO (R3) : détection complète d'une zone")

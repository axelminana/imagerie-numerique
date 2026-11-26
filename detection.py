"""Étape 3 : détection des usagers avec YOLO (responsable : R3).

Détecte les objets sur les images conservées par le filtre, les regroupe dans
les catégories du projet, marque les personnes qui conduisent un deux-roues
(is_rider), puis appelle parked.py pour marquer les véhicules stationnés.

Produit detections.csv (colonnes : config.DETECTION_COLUMNS) et, pour la
soutenance, quelques images annotées dans exemples_detection/.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import config
from utils import resolve_device as _resolve_device

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


def _box(detection: dict) -> Box:
    """Boîte englobante d'une détection."""
    return detection["x1"], detection["y1"], detection["x2"], detection["y2"]


def mark_riders(detections: list[dict],
                threshold: float = config.RIDER_OVERLAP_THRESHOLD) -> list[dict]:
    """Met is_rider à True pour les personnes qui conduisent un deux-roues.

    Une personne est candidate pour un deux-roues si covered_fraction(personne,
    deux-roues) dépasse threshold et si son centre horizontal tombe dans la
    boîte du deux-roues. Les associations sont faites de la plus forte à la plus
    faible, chaque personne au plus une fois, dans la limite de
    config.RIDER_CAPACITY (un conducteur par vélo, deux par moto).
    Les détections sont celles d'une seule image ; la liste est modifiée et retournée.
    """
    for det in detections:
        det["is_rider"] = False

    persons = [i for i, d in enumerate(detections) if d["category"] == "pietons"]
    vehicles = [i for i, d in enumerate(detections)
                if d["category"] in config.TWO_WHEELER_CATEGORIES]

    pairs = []
    for p in persons:
        person_box = _box(detections[p])
        center_x = (person_box[0] + person_box[2]) / 2
        for v in vehicles:
            vehicle_box = _box(detections[v])
            fraction = covered_fraction(person_box, vehicle_box)
            if fraction >= threshold and vehicle_box[0] <= center_x <= vehicle_box[2]:
                pairs.append((fraction, p, v))

    capacity = {v: config.RIDER_CAPACITY.get(detections[v]["category"], 1) for v in vehicles}
    for fraction, p, v in sorted(pairs, reverse=True):
        if not detections[p]["is_rider"] and capacity[v] > 0:
            detections[p]["is_rider"] = True
            capacity[v] -= 1
    return detections


def _load_model(model_path: str):
    """Charge un modèle YOLO (les poids officiels sont téléchargés au premier appel)."""
    from ultralytics import YOLO
    return YOLO(model_path)


def run_yolo(image_paths: list[Path], model_path: str, conf: float, imgsz: int,
             device: str, examples_dir: Path | None = None,
             n_examples: int = config.N_DETECTION_EXAMPLES) -> list[dict]:
    """Lance YOLO par lots et retourne une détection par objet retenu.

    Seules les classes de config.COCO_TO_CATEGORY sont demandées au modèle.
    Chaque détection contient les colonnes de detections.csv (sauf is_rider et
    parked, remplies ensuite) ainsi que image_width et image_height, utiles au
    tri des véhicules stationnés.
    Si examples_dir est donné, les n_examples premières images y sont
    enregistrées avec leurs boîtes dessinées.
    """
    from tqdm import tqdm

    model = _load_model(model_path)
    device = _resolve_device(device)
    class_ids = [i for i, name in model.names.items() if name in config.COCO_TO_CATEGORY]
    if examples_dir is not None:
        examples_dir.mkdir(parents=True, exist_ok=True)

    detections: list[dict] = []
    n_saved = 0
    batches = [image_paths[i:i + config.YOLO_BATCH_SIZE]
               for i in range(0, len(image_paths), config.YOLO_BATCH_SIZE)]
    for batch in tqdm(batches, desc="YOLO", unit="lot"):
        results = model.predict(source=[str(p) for p in batch], conf=conf, imgsz=imgsz,
                                device=device, classes=class_ids, verbose=False)
        for path, result in zip(batch, results):
            height, width = result.orig_shape[:2]
            boxes = result.boxes
            for xyxy, cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(),
                                        boxes.conf.tolist()):
                coco_class = model.names[int(cls)]
                category = coco_to_category(coco_class)
                if category is None:
                    continue
                detections.append({
                    "image_id": path.stem, "coco_class": coco_class, "category": category,
                    "conf": round(float(score), 4),
                    "x1": round(xyxy[0], 1), "y1": round(xyxy[1], 1),
                    "x2": round(xyxy[2], 1), "y2": round(xyxy[3], 1),
                    "image_width": width, "image_height": height,
                })
            if examples_dir is not None and n_saved < n_examples:
                import cv2
                cv2.imwrite(str(examples_dir / f"{path.stem}.jpg"), result.plot())
                n_saved += 1
    return detections


def _read_csv(path: Path) -> list[dict]:
    """Lit un CSV en liste de dictionnaires (toutes les valeurs en texte)."""
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def images_to_process(zone_dir: Path) -> list[Path]:
    """Chemins des images conservées par le filtre (keep vrai dans filter.csv)."""
    filter_path = zone_dir / config.FILTER_FILE
    metadata_path = zone_dir / config.METADATA_FILE
    for path, step in ((metadata_path, "download"), (filter_path, "filter")):
        if not path.exists():
            raise RuntimeError(f"{path} introuvable : lancez d'abord l'étape {step}.")

    kept = {row["image_id"] for row in _read_csv(filter_path)
            if row["keep"].strip().lower() == "true"}
    paths = []
    for row in _read_csv(metadata_path):
        if row["image_id"] in kept and row["path"]:
            image_path = zone_dir / row["path"]
            if image_path.exists():
                paths.append(image_path)
            else:
                log.warning("Image absente du disque : %s", image_path)
    return paths


def write_detections(detections: list[dict], path: Path) -> None:
    """Écrit detections.csv avec exactement les colonnes de config.DETECTION_COLUMNS."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.DETECTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detections)


def detect(zone_dir: Path, model_path: str = config.DEFAULT_DETECTOR,
           conf: float = config.DEFAULT_CONF, imgsz: int = config.DEFAULT_IMGSZ,
           device: str = "cuda",
           parked_strategy: str = config.DEFAULT_PARKED_STRATEGY) -> Path:
    """Exécute l'étape 3 sur les images gardées dans filter.csv.

    Enchaîne run_yolo, mark_riders (image par image), puis parked.mark_parked.
    Écrit detections.csv dans zone_dir et retourne son chemin.
    """
    import parked

    zone_dir = Path(zone_dir)
    image_paths = images_to_process(zone_dir)
    log.info("Détection sur %d image(s) avec %s (conf=%s, imgsz=%s)",
             len(image_paths), model_path, conf, imgsz)

    detections = run_yolo(image_paths, model_path, conf, imgsz, device,
                          examples_dir=zone_dir / config.EXAMPLES_DIR)

    by_image: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        by_image[det["image_id"]].append(det)
    for image_detections in by_image.values():
        mark_riders(image_detections)

    detections = parked.mark_parked(detections, zone_dir, strategy=parked_strategy, device=device)

    output = zone_dir / config.DETECTIONS_FILE
    write_detections(detections, output)
    n_riders = sum(d["is_rider"] for d in detections)
    log.info("%d objet(s) détecté(s) dont %d conducteur(s) de deux-roues ; résultats dans %s",
             len(detections), n_riders, output)
    return output

"""Tests de la détection, du rapport et du pipeline complet, avec un faux modèle YOLO.

Lancer : python -m pytest tests/test_detection_report.py -v
"""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import detection  # noqa: E402
import main  # noqa: E402
import report  # noqa: E402


def det(category, box, image_id="1", conf=0.9):
    """Détection minimale pour mark_riders."""
    x1, y1, x2, y2 = box
    return {"image_id": image_id, "category": category, "conf": conf,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2}


# ---------------------------------------------------------------------------
# Dédoublonnage des conducteurs
# ---------------------------------------------------------------------------
def test_cycliste_et_pieton():
    dets = [det("deux_roues", (100, 150, 200, 250)),     # vélo
            det("pietons", (120, 80, 180, 220)),         # cycliste sur le vélo
            det("pietons", (400, 80, 450, 220))]         # piéton plus loin
    detection.mark_riders(dets)
    assert [d["is_rider"] for d in dets] == [False, True, False]


def test_un_seul_conducteur_par_velo():
    dets = [det("deux_roues", (100, 150, 200, 250)),
            det("pietons", (120, 100, 180, 240)),        # très recouvert : conducteur
            det("pietons", (130, 90, 190, 230))]         # recouvert mais moins
    detection.mark_riders(dets)
    assert [d["is_rider"] for d in dets[1:]] == [True, False]


def test_deux_personnes_sur_une_moto():
    dets = [det("deux_roues_motorises", (100, 150, 260, 250)),
            det("pietons", (120, 100, 180, 240)),
            det("pietons", (170, 100, 230, 240))]
    detection.mark_riders(dets)
    assert [d["is_rider"] for d in dets[1:]] == [True, True]


def test_personne_a_cote_du_velo_pas_conductrice():
    # boîtes qui se touchent mais centre de la personne hors du vélo
    dets = [det("deux_roues", (100, 150, 200, 250)), det("pietons", (180, 100, 260, 240))]
    detection.mark_riders(dets)
    assert dets[1]["is_rider"] is False


# ---------------------------------------------------------------------------
# Faux modèle YOLO
# ---------------------------------------------------------------------------
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 16: "dog"}


class FakeYOLO:
    """Imite l'interface d'ultralytics.YOLO : names et predict()."""

    names = COCO_NAMES

    def __init__(self, boxes_by_image):
        self.boxes_by_image = boxes_by_image
        self.calls = []

    def predict(self, source, conf, imgsz, device, classes, verbose):
        self.calls.append({"n": len(source), "classes": classes, "device": device})
        results = []
        for path in source:
            rows = self.boxes_by_image.get(Path(path).stem, [])
            xyxy = np.array([r[1] for r in rows], dtype=float).reshape(-1, 4)
            results.append(SimpleNamespace(
                orig_shape=(1536, 2048),
                boxes=SimpleNamespace(xyxy=xyxy,
                                      cls=np.array([r[0] for r in rows], dtype=float),
                                      conf=np.array([r[2] for r in rows], dtype=float)),
                plot=lambda: np.zeros((10, 10, 3), dtype=np.uint8)))
        return results


BOXES = {
    "1": [(2, (500, 600, 900, 900), 0.91),       # voiture
          (1, (100, 700, 200, 850), 0.80),       # vélo
          (0, (110, 600, 190, 840), 0.85),       # cycliste
          (0, (1500, 600, 1560, 800), 0.70)],    # piéton
    "2": [(7, (300, 400, 900, 1000), 0.60),      # camion
          (16, (10, 10, 50, 50), 0.99)],         # chien : ignoré
    "3": [],                                     # image vide
}


def make_zone(root, image_ids=("1", "2", "3"), rejected=()):
    """Crée un dossier de zone avec metadata.csv et des images factices."""
    zone = root / "data" / "essai"
    (zone / "images").mkdir(parents=True)
    with open(zone / config.METADATA_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.METADATA_COLUMNS)
        writer.writeheader()
        for i in image_ids:
            (zone / "images" / f"{i}.jpg").write_bytes(b"jpg")
            writer.writerow({"image_id": i, "sequence_id": "s", "lat": 45.8, "lon": 3.1,
                             "distance_m": 10, "captured_at": "", "compass_angle": "",
                             "is_pano": False, "selected": True, "path": f"images/{i}.jpg"})
        writer.writerow({"image_id": "99", "sequence_id": "s", "lat": 45.8, "lon": 3.1,
                         "distance_m": 150, "captured_at": "", "compass_angle": "",
                         "is_pano": "", "selected": False, "path": ""})
    return zone


@pytest.fixture
def fake_yolo(monkeypatch):
    model = FakeYOLO(BOXES)
    monkeypatch.setattr(detection, "_load_model", lambda path: model)
    monkeypatch.setattr(detection, "_resolve_device", lambda device: "cpu")
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imwrite=lambda path, img: True))
    return model


def test_run_yolo_regroupe_les_classes(tmp_path, fake_yolo):
    paths = [tmp_path / f"{i}.jpg" for i in ("1", "2", "3")]
    dets = detection.run_yolo(paths, "faux.pt", conf=0.4, imgsz=1280, device="cpu")
    assert [d["category"] for d in dets] == [
        "vehicules_legers", "deux_roues", "pietons", "pietons", "camions"]
    assert dets[0]["image_id"] == "1" and dets[0]["image_width"] == 2048
    assert 16 not in fake_yolo.calls[0]["classes"]           # le chien n'est pas demandé


def read_csv(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Pipeline complet (filtre "none", détection, rapport) lancé par main.py
# ---------------------------------------------------------------------------
def test_pipeline_complet(tmp_path, monkeypatch, fake_yolo):
    monkeypatch.chdir(tmp_path)
    zone = make_zone(tmp_path)

    code = main.main(["-lat", "45.8", "-lon", "3.1", "--zone", "essai", "--steps",
                      "filter,detect,report", "--filter", "none", "--parked", "none",
                      "--device", "cpu"])
    assert code == 0

    detections = read_csv(zone / config.DETECTIONS_FILE)
    assert list(detections[0]) == config.DETECTION_COLUMNS
    riders = [d for d in detections if d["is_rider"] == "True"]
    assert len(riders) == 1 and riders[0]["image_id"] == "1"
    car = next(d for d in detections if d["category"] == "vehicules_legers")
    assert car["parked"] == "False" and car["parked_method"] == "none"

    rows = {r["image"]: r for r in read_csv(zone / config.REPORT_FILE)}
    assert list(next(iter(rows.values()))) == config.REPORT_COLUMNS
    assert set(rows) == {"1.jpg", "2.jpg", "3.jpg"}           # l'image 99 n'est pas sélectionnée
    assert rows["1.jpg"]["pietons"] == "1"                     # le cycliste n'est pas un piéton
    assert rows["1.jpg"]["deux_roues"] == "1"
    assert rows["1.jpg"]["vehicules_legers"] == "1"
    assert rows["2.jpg"]["camions"] == "1"
    assert all(rows["3.jpg"][c] == "0" for c in config.CATEGORIES)   # image vide : zéros


# ---------------------------------------------------------------------------
# Rapport : images rejetées et véhicules stationnés
# ---------------------------------------------------------------------------
def write_rows(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_rapport_exclut_rejetees_et_stationnes(tmp_path):
    zone = make_zone(tmp_path, image_ids=("1", "2"))
    write_rows(zone / config.FILTER_FILE, config.FILTER_COLUMNS, [
        {"image_id": "1", "method": "clip", "answer": "", "score": 0.9, "keep": True},
        {"image_id": "2", "method": "clip", "answer": "", "score": 0.1, "keep": False}])
    base = {"coco_class": "car", "category": "vehicules_legers", "conf": 0.9,
            "x1": 0, "y1": 0, "x2": 1, "y2": 1, "is_rider": False, "parked_method": "geometry"}
    write_rows(zone / config.DETECTIONS_FILE, config.DETECTION_COLUMNS, [
        {**base, "image_id": "1", "parked": False},
        {**base, "image_id": "1", "parked": True},
        {**base, "image_id": "2", "parked": False}])   # image rejetée par le filtre

    output = report.build_report(zone, zone / "report.csv")
    rows = read_csv(output)
    assert [r["image"] for r in rows] == ["1.jpg"]
    assert rows[0]["vehicules_legers"] == "1"

    rows = read_csv(report.build_report(zone, zone / "report_tous.csv", keep_parked=True))
    assert rows[0]["vehicules_legers"] == "2"


def test_rapport_sans_detection(tmp_path):
    zone = make_zone(tmp_path, image_ids=("1",))
    write_rows(zone / config.FILTER_FILE, config.FILTER_COLUMNS,
               [{"image_id": "1", "method": "none", "answer": "", "score": "", "keep": True}])
    write_rows(zone / config.DETECTIONS_FILE, config.DETECTION_COLUMNS, [])
    rows = read_csv(report.build_report(zone, zone / "report.csv"))
    assert rows == [{"image": "1.jpg", **{c: "0" for c in config.CATEGORIES}}]


def test_etape_manquante_message_clair(tmp_path):
    zone = make_zone(tmp_path)
    with pytest.raises(RuntimeError, match="lancez d'abord l'étape filter"):
        detection.images_to_process(zone)

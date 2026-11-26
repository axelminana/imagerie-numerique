"""Tests de la préparation et du regroupement des annotations."""
import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import prepare_annotation as pa  # noqa: E402


def make_zone(root, name, n):
    zone = root / "data" / name
    (zone / "images").mkdir(parents=True)
    with open(zone / config.METADATA_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.METADATA_COLUMNS)
        writer.writeheader()
        for i in range(n):
            image_id = f"{name[0]}{i:03d}"
            (zone / "images" / f"{image_id}.jpg").write_bytes(b"jpg")
            writer.writerow({"image_id": image_id, "sequence_id": "s", "lat": 0, "lon": 0,
                             "distance_m": 0, "captured_at": "", "compass_angle": "",
                             "is_pano": False, "selected": True, "path": f"images/{image_id}.jpg"})
        writer.writerow({"image_id": "pas_retenue", "sequence_id": "s", "lat": 0, "lon": 0,
                         "distance_m": 0, "captured_at": "", "compass_angle": "",
                         "is_pano": "", "selected": False, "path": ""})
    return zone


def read_sheet(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def test_repartition_en_trois(tmp_path):
    zones = [make_zone(tmp_path, "carrefour", 75), make_zone(tmp_path, "dam", 75)]
    out = tmp_path / "annotation"
    rows = pa.prepare(zones, ["A", "B", "C"], out, n_doubles=21)

    sheets = {name: read_sheet(out / f"feuille_{name}.csv") for name in "ABC"}
    # 150 images + 21 doubles = 171 lignes, 57 par personne
    assert sum(len(s) for s in sheets.values()) == 171
    assert all(len(s) == 57 for s in sheets.values())
    # chaque image une fois, sauf les doubles deux fois, et jamais deux fois par la même personne
    counts = Counter(r["image_id"] for r in rows)
    assert len(counts) == 150 and sum(v == 2 for v in counts.values()) == 21
    for image_id, n in counts.items():
        if n == 2:
            assert len({r["annotateur"] for r in rows if r["image_id"] == image_id}) == 2
    # chaque personne a les deux zones et les deux moitiés
    for sheet in sheets.values():
        assert {r["zone"] for r in sheet} == {"carrefour", "dam"}
        assert {r["split"] for r in sheet} == {"reglage", "test"}
    # moitié réglage / moitié test dans chaque zone
    first = {r["image_id"]: r for r in rows}
    for zone in ("carrefour", "dam"):
        splits = Counter(r["split"] for r in first.values() if r["zone"] == zone)
        assert splits == {"reglage": 37, "test": 38}
    # images copiées et consigne écrite
    assert all((out / r["fichier"]).exists() for s in sheets.values() for r in s)
    assert "keep" in (out / "consigne.txt").read_text(encoding="utf-8")


def test_reproductible(tmp_path):
    zones = [make_zone(tmp_path, "carrefour", 30)]
    a = pa.prepare(zones, ["A", "B", "C"], tmp_path / "x", n_doubles=6)
    b = pa.prepare(zones, ["A", "B", "C"], tmp_path / "y", n_doubles=6)
    key = lambda r: (r["image_id"], r["annotateur"], r["split"])
    assert sorted(map(key, a)) == sorted(map(key, b))


def test_dossier_non_vide_refuse(tmp_path):
    zones = [make_zone(tmp_path, "carrefour", 10)]
    pa.prepare(zones, ["A", "B"], tmp_path / "annotation", n_doubles=2)
    with pytest.raises(RuntimeError, match="existe déjà"):
        pa.prepare(zones, ["A", "B"], tmp_path / "annotation", n_doubles=2)


def test_regroupement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    zones = [make_zone(tmp_path, "carrefour", 6)]
    out = tmp_path / "annotation"
    pa.prepare(zones, ["A", "B"], out, n_doubles=2)
    # A remplit sa feuille dans Excel (séparateur ;) ; B l'exporte de Google Sheets (séparateur ,)
    for name, delimiter in (("A", ";"), ("B", ",")):
        rows = read_sheet(out / f"feuille_{name}.csv")
        for row in rows:
            row.update({"keep": "oui", **{c: "1" for c in config.CATEGORIES}, "stationnes": "2"})
        with open(out / f"feuille_{name}.csv", "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=pa.SHEET_COLUMNS, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

    path = pa.merge(out, tmp_path / "annotations" / "annotations.csv")
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == config.ANNOTATION_COLUMNS
        merged = list(reader)
    assert len(merged) == 8 and all(r["keep"] == "oui" and r["stationnes"] == "2" for r in merged)
    assert {r["annotateur"] for r in merged} == {"A", "B"}

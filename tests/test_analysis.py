"""Tests de l'analyse statistique et des graphiques (étape 5), sur des zones simulées.

Lancer : python -m pytest tests/test_analysis.py -v
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis  # noqa: E402
import config  # noqa: E402


def make_zone(root, name, images, kept=None, year=2023):
    """Zone simulée. images : liste de (sequence_id, {catégorie: nombre})."""
    zone = root / name
    zone.mkdir(parents=True)
    kept = set(range(len(images))) if kept is None else set(kept)
    with open(zone / config.METADATA_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.METADATA_COLUMNS)
        writer.writeheader()
        for i, (seq, _) in enumerate(images):
            writer.writerow({"image_id": str(i), "sequence_id": seq, "lat": 45.8 + i * 1e-4,
                             "lon": 3.1 + i * 1e-4, "distance_m": i, "compass_angle": "",
                             "captured_at": f"{year + i % 2}-06-01T10:00:00+00:00",
                             "is_pano": False, "selected": True, "path": f"images/{i}.jpg"})
        writer.writerow({"image_id": "999", "sequence_id": "x", "lat": 45.81, "lon": 3.11,
                         "distance_m": 180, "compass_angle": "", "captured_at": "",
                         "is_pano": "", "selected": False, "path": ""})
    with open(zone / config.FILTER_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.FILTER_COLUMNS)
        writer.writeheader()
        for i in range(len(images)):
            writer.writerow({"image_id": str(i), "method": "clip", "answer": "",
                             "score": 0.9, "keep": i in kept})
    with open(zone / config.REPORT_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.REPORT_COLUMNS)
        writer.writeheader()
        for i, (_, counts) in enumerate(images):
            if i in kept:
                writer.writerow({"image": f"{i}.jpg",
                                 **{c: counts.get(c, 0) for c in config.CATEGORIES}})
    return zone


# zone A : 4 images sur 2 séquences
IMAGES_A = [("s1", {"pietons": 2, "vehicules_legers": 1}),
            ("s1", {"vehicules_legers": 3}),
            ("s2", {"pietons": 1, "deux_roues": 1}),
            ("s2", {})]


def test_load_zone(tmp_path):
    zone = make_zone(tmp_path, "a", IMAGES_A)
    data = analysis.load_zone(zone / config.REPORT_FILE, zone / config.METADATA_FILE)
    assert list(data["image_id"]) == ["0", "1", "2", "3"]
    assert list(data["sequence_id"]) == ["s1", "s1", "s2", "s2"]
    assert data["vehicules_legers"].sum() == 4


def test_indicateurs(tmp_path):
    zone = make_zone(tmp_path, "a", IMAGES_A)
    table = analysis.compute_indicators(zone / config.REPORT_FILE, zone / config.METADATA_FILE, "a")
    assert list(table.columns) == config.STATS_COLUMNS
    stats = table.set_index("category")
    assert stats.loc["pietons", "count"] == 3
    assert stats.loc["pietons", "share"] == pytest.approx(3 / 8)          # 8 objets au total
    assert stats.loc["vehicules_legers", "share"] == pytest.approx(0.5)
    assert stats.loc["pietons", "presence_rate"] == pytest.approx(0.5)     # 2 images sur 4
    assert stats.loc["vehicules_legers", "mean_per_image"] == pytest.approx(1.0)
    assert stats["share"].sum() == pytest.approx(1)


def test_zone_sans_objet(tmp_path):
    zone = make_zone(tmp_path, "vide", [("s1", {}), ("s2", {})])
    stats = analysis.compute_indicators(zone / config.REPORT_FILE, zone / config.METADATA_FILE, "v")
    assert stats["count"].sum() == 0 and stats["share"].isna().all()


def many_sequences(n_seq, car_share, rng):
    """n_seq séquences de 3 images ; chaque objet est une voiture avec la probabilité car_share."""
    images = []
    for s in range(n_seq):
        for _ in range(3):
            n = rng.integers(0, 6)
            cars = int(rng.binomial(n, car_share))
            images.append((f"s{s}", {"vehicules_legers": cars, "pietons": int(n - cars)}))
    return images


def test_bootstrap_contient_l_estimation_et_reproductible(tmp_path):
    rng = np.random.default_rng(1)
    zone = make_zone(tmp_path, "a", many_sequences(40, 0.7, rng))
    data = analysis.load_zone(zone / config.REPORT_FILE, zone / config.METADATA_FILE)
    stats = analysis.zone_stats(data, "a").set_index("category")
    cars = stats.loc["vehicules_legers"]
    assert cars["share_low"] < cars["share"] < cars["share_high"]
    assert 0.55 < cars["share"] < 0.85
    # catégorie absente : intervalle réduit à 0
    assert stats.loc["bus", "share_low"] == 0 and stats.loc["bus", "share_high"] == 0
    # même graine : mêmes bornes
    again = analysis.bootstrap_shares(data).set_index("category")
    assert again.loc["vehicules_legers", "share_low"] == cars["share_low"]


def test_bootstrap_tire_les_sequences(tmp_path):
    # deux séquences très différentes : un tirage par séquence donne des parts extrêmes
    images = [("s1", {"pietons": 10})] * 5 + [("s2", {"vehicules_legers": 10})] * 5
    zone = make_zone(tmp_path, "a", images)
    data = analysis.load_zone(zone / config.REPORT_FILE, zone / config.METADATA_FILE)
    ci = analysis.bootstrap_shares(data, n_boot=2000).set_index("category")
    # par séquence, la part des piétons vaut 0, 0,5 ou 1 : l'intervalle couvre presque tout
    assert ci.loc["pietons", "share_low"] == 0
    assert ci.loc["pietons", "share_high"] == 1


def test_bootstrap_une_seule_sequence(tmp_path):
    zone = make_zone(tmp_path, "a", [("s1", {"pietons": 1}), ("s1", {"bus": 1})])
    data = analysis.load_zone(zone / config.REPORT_FILE, zone / config.METADATA_FILE)
    assert analysis.bootstrap_shares(data)["share_low"].isna().all()


def test_comparaison_des_zones(tmp_path):
    rng = np.random.default_rng(2)
    za = make_zone(tmp_path, "a", many_sequences(40, 0.9, rng))
    zb = make_zone(tmp_path, "b", many_sequences(40, 0.2, rng))
    stats = pd.concat([analysis.compute_indicators(z / config.REPORT_FILE, z / config.METADATA_FILE, n)
                       for z, n in ((za, "a"), (zb, "b"))])
    result = analysis.compare_zones(stats)
    assert result["ok"] and result["p_value"] < 1e-6 and result["cramer_v"] > 0.5
    assert list(result["table"].columns) == ["pietons", "vehicules_legers"]   # colonnes vides retirées
    assert "V de Cramér" in analysis.format_comparison(result)


def test_comparaison_une_seule_zone(tmp_path):
    zone = make_zone(tmp_path, "a", IMAGES_A)
    stats = analysis.compute_indicators(zone / config.REPORT_FILE, zone / config.METADATA_FILE, "a")
    assert analysis.compare_zones(stats)["ok"] is False


def test_analyse_complete(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(3)
    za = make_zone(tmp_path, "carrefour", many_sequences(15, 0.9, rng), kept=range(40))
    zb = make_zone(tmp_path, "dam", many_sequences(15, 0.2, rng))
    out = tmp_path / "figures"

    code = analysis.main(["--zones", str(za), str(zb), "--names", "Carrefour", "Dam",
                          "--out", str(out)])
    assert code == 0

    stats = pd.read_csv(out / config.STATS_FILE)
    assert list(stats.columns) == config.STATS_COLUMNS
    assert set(stats["zone"]) == {"Carrefour", "Dam"} and len(stats) == 12
    assert "khi-deux" in (out / "comparaison.txt").read_text(encoding="utf-8")
    for name in ("parts_modales.png", "comparaison_zones.png", "carte_Carrefour.png",
                 "carte_Dam.png", "dates.png"):
        assert (out / name).stat().st_size > 5000, name


def test_rapport_manquant_message_clair(tmp_path):
    zone = tmp_path / "vide"
    zone.mkdir()
    assert analysis.main(["--zones", str(zone), "--out", str(tmp_path / "f")]) == 1

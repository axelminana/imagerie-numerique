"""Tests des fonctions utilitaires déjà écrites (lancer : python -m pytest)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection import coco_to_category, covered_fraction  # noqa: E402
from filtering import parse_yes_no  # noqa: E402
from mapillary_client import bbox_from_radius, haversine_m  # noqa: E402

LAT, LON = 45.800413, 3.108813  # carrefour des pistes


def test_haversine_zero():
    assert haversine_m(LAT, LON, LAT, LON) == 0


def test_haversine_un_degre_de_latitude():
    # un degré de latitude fait environ 111 km
    assert haversine_m(45.0, 3.0, 46.0, 3.0) == pytest.approx(111_195, rel=1e-3)


def test_bbox_contient_le_cercle():
    west, south, east, north = bbox_from_radius(LAT, LON, 200)
    # chaque côté de la boîte est à environ 200 m du centre
    assert haversine_m(LAT, LON, north, LON) == pytest.approx(200, rel=0.01)
    assert haversine_m(LAT, LON, south, LON) == pytest.approx(200, rel=0.01)
    assert haversine_m(LAT, LON, LAT, east) == pytest.approx(200, rel=0.01)
    assert haversine_m(LAT, LON, LAT, west) == pytest.approx(200, rel=0.01)


def test_covered_fraction():
    person = (10, 0, 20, 40)
    assert covered_fraction(person, (0, 20, 30, 50)) == pytest.approx(0.5)
    assert covered_fraction(person, (100, 100, 120, 120)) == 0
    assert covered_fraction(person, (0, 0, 50, 50)) == 1


def test_coco_to_category():
    assert coco_to_category("car") == "vehicules_legers"
    assert coco_to_category("person") == "pietons"
    assert coco_to_category("dog") is None


@pytest.mark.parametrize("answer,expected", [
    ("yes", True), (" Yes, it is", True), ("no", False), ("a street", False), ("", False),
])
def test_parse_yes_no(answer, expected):
    assert parse_yes_no(answer) is expected

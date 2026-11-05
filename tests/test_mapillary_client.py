"""Tests de la collecte Mapillary, sans accès réseau (réponses simulées).

Lancer : python -m pytest tests/test_mapillary_client.py -v
"""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import mapillary_client as mc  # noqa: E402

LAT, LON = 45.800413, 3.108813      # carrefour des pistes
DEG_PER_M = 1 / 111_195             # degrés de latitude pour un mètre


def make_image(image_id, seq, north_m, t_ms, is_pano=False):
    """Image simulée située north_m mètres au nord du centre."""
    return {"image_id": image_id, "sequence_id": seq, "lat": LAT + north_m * DEG_PER_M,
            "lon": LON, "distance_m": abs(north_m), "captured_at_ms": t_ms,
            "compass_angle": 90.0, "is_pano": is_pano}


# ---------------------------------------------------------------------------
# Sous-échantillonnage
# ---------------------------------------------------------------------------
def test_subsample_garde_une_image_tous_les_15_m():
    # images tous les 6 m : on garde 0 m, 18 m et 36 m
    images = [make_image(str(i), "s1", 6 * i, 1000 * i) for i in range(7)]
    kept = mc.subsample_by_spacing(images, 15)
    assert [img["image_id"] for img in kept] == ["0", "3", "6"]


def test_subsample_sequences_independantes():
    images = ([make_image(f"a{i}", "s1", 6 * i, i) for i in range(4)]
              + [make_image(f"b{i}", "s2", 6 * i, i) for i in range(4)])
    kept_ids = {img["image_id"] for img in mc.subsample_by_spacing(images, 15)}
    assert kept_ids == {"a0", "a3", "b0", "b3"}


def test_subsample_suit_l_ordre_chronologique():
    # mêmes positions mais données dans le désordre
    images = [make_image(str(i), "s1", 6 * i, 1000 * i) for i in range(7)][::-1]
    kept = mc.subsample_by_spacing(images, 15)
    assert [img["image_id"] for img in kept] == ["0", "3", "6"]


def test_subsample_espacement_nul_garde_tout():
    images = [make_image(str(i), "s1", 1 * i, i) for i in range(5)]
    assert len(mc.subsample_by_spacing(images, 0)) == 5


# ---------------------------------------------------------------------------
# Lecture des tuiles
# ---------------------------------------------------------------------------
def feature(image_id, north_m, seq="s1", is_pano=False):
    return {"geometry": {"coordinates": [LON, LAT + north_m * DEG_PER_M]},
            "properties": {"id": image_id, "sequence_id": seq, "captured_at": 1_700_000_000_000,
                           "compass_angle": 12.5, "is_pano": is_pano}}


def test_list_candidate_images_filtre_le_cercle(monkeypatch):
    features = [feature(1, 50), feature(2, 150), feature(3, 260), feature(1, 50)]
    monkeypatch.setattr(mc, "_get", lambda *a, **k: SimpleNamespace(status_code=200, content=b"x"))
    monkeypatch.setattr(mc, "_decode_tile", lambda content, tile: features)

    images = mc.list_candidate_images(LAT, LON, 200, token="t", source="tiles")

    # l'image à 260 m est hors du cercle, le doublon de l'image 1 est ignoré
    assert [img["image_id"] for img in images] == ["1", "2"]
    assert images[0]["distance_m"] == pytest.approx(50, abs=0.5)


def test_token_refuse_donne_un_message_clair(monkeypatch):
    monkeypatch.setattr(mc, "_get", lambda *a, **k: SimpleNamespace(status_code=401, content=b""))
    with pytest.raises(RuntimeError, match="refuse le token"):
        mc.list_candidate_images(LAT, LON, 200, token="mauvais", source="tiles")


# ---------------------------------------------------------------------------
# Requêtes : nouvelles tentatives
# ---------------------------------------------------------------------------
class FakeSession:
    """Session qui renvoie successivement les codes HTTP donnés."""

    def __init__(self, codes):
        self.codes = list(codes)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return SimpleNamespace(status_code=self.codes.pop(0))


def test_get_reessaie_apres_une_erreur_temporaire(monkeypatch):
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    session = FakeSession([503, 200])
    assert mc._get("https://exemple", session).status_code == 200
    assert session.calls == 2


def test_get_abandonne_apres_trois_tentatives(monkeypatch):
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="tentatives"):
        mc._get("https://exemple", FakeSession([503] * config.REQUEST_RETRIES))


def test_download_image_ne_retelecharge_pas(tmp_path, monkeypatch):
    path = tmp_path / "images" / "1.jpg"
    path.parent.mkdir()
    path.write_bytes(b"deja la")
    monkeypatch.setattr(mc, "_get", lambda *a, **k: pytest.fail("aucune requête attendue"))
    assert mc.download_image("https://exemple", path) is True


# ---------------------------------------------------------------------------
# Collecte complète
# ---------------------------------------------------------------------------
def test_collect_ecrit_metadata(tmp_path, monkeypatch):
    candidates = [
        make_image("1", "s1", 0, 1_700_000_000_000),
        make_image("2", "s1", 5, 1_700_000_001_000),    # trop proche de 1 : écartée
        make_image("3", "s1", 20, 1_700_000_002_000),
        make_image("4", "s2", 40, 1_700_000_000_000, is_pano=True),   # panorama connu
        make_image("5", "s3", 60, None, is_pano=None),  # panorama révélé par l'API Graph
        make_image("6", "s4", 80, 1_700_000_000_000),   # API Graph sans vignette
    ]
    monkeypatch.setenv(config.MAPILLARY_TOKEN_ENV, "token-de-test")
    monkeypatch.setattr(mc, "list_candidate_images", lambda *a, **k: candidates)

    def fake_info(image_id, token, session=None):
        if image_id == "6":
            return None
        return {"thumb_2048_url": f"https://exemple/{image_id}.jpg",
                "is_pano": image_id == "5", "captured_at": 1_700_000_000_000}

    def fake_download(url, path, session=None):
        path.write_bytes(b"jpg")
        return True

    monkeypatch.setattr(mc, "fetch_image_info", fake_info)
    monkeypatch.setattr(mc, "download_image", fake_download)

    metadata_path = mc.collect(LAT, LON, 200, tmp_path, spacing_m=15)

    with open(metadata_path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == config.METADATA_COLUMNS
        rows = {row["image_id"]: row for row in reader}

    assert set(rows) == {"1", "2", "3", "4", "5", "6"}          # toutes les candidates
    selected = {i for i, row in rows.items() if row["selected"] == "True"}
    assert selected == {"1", "3"}
    assert rows["1"]["path"] == "images/1.jpg"
    assert (tmp_path / "images" / "1.jpg").exists()
    assert rows["5"]["is_pano"] == "True"
    assert rows["1"]["captured_at"].startswith("2023-11-14")


def test_collect_max_images_reproductible(tmp_path, monkeypatch):
    candidates = [make_image(str(i), f"s{i}", i, i) for i in range(20)]
    monkeypatch.setenv(config.MAPILLARY_TOKEN_ENV, "token-de-test")
    monkeypatch.setattr(mc, "list_candidate_images", lambda *a, **k: candidates)
    monkeypatch.setattr(mc, "fetch_image_info",
                        lambda i, t, session=None: {"thumb_2048_url": "u", "is_pano": False})
    monkeypatch.setattr(mc, "download_image", lambda url, path, session=None: True)

    def selected_ids(folder):
        with open(mc.collect(LAT, LON, 200, folder, spacing_m=0, max_images=5),
                  encoding="utf-8") as handle:
            return {r["image_id"] for r in csv.DictReader(handle) if r["selected"] == "True"}

    first = selected_ids(tmp_path / "a")
    assert len(first) == 5
    assert selected_ids(tmp_path / "b") == first


def test_token_absent(monkeypatch):
    monkeypatch.delenv(config.MAPILLARY_TOKEN_ENV, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    with pytest.raises(RuntimeError, match="introuvable"):
        mc.get_token()


def test_decode_tile_reelle():
    """Décode une vraie tuile vectorielle encodée localement (point au centre de la tuile)."""
    import mapbox_vector_tile
    import mercantile

    tile = mercantile.tile(LON, LAT, config.MAPILLARY_TILE_ZOOM)
    content = mapbox_vector_tile.encode([{
        "name": "image",
        "features": [{"geometry": "POINT(2048 2048)",
                      "properties": {"id": 42, "sequence_id": "abc", "is_pano": False}}],
    }])
    features = mc._decode_tile(content, tile)
    bounds = mercantile.bounds(tile)

    assert len(features) == 1
    lon, lat = features[0]["geometry"]["coordinates"][:2]
    assert bounds.west < lon < bounds.east and bounds.south < lat < bounds.north
    assert features[0]["properties"]["id"] == 42


def test_decode_tile_compressee_gzip():
    """Une tuile compressée en gzip est décompressée avant le décodage."""
    import gzip

    import mapbox_vector_tile
    import mercantile

    tile = mercantile.tile(LON, LAT, config.MAPILLARY_TILE_ZOOM)
    content = gzip.compress(mapbox_vector_tile.encode([{
        "name": "image",
        "features": [{"geometry": "POINT(2048 2048)", "properties": {"id": 7}}],
    }]))
    assert mc._decode_tile(content, tile)[0]["properties"]["id"] == 7


def test_reponse_invalide_message_explicite(monkeypatch):
    """Une réponse qui n'est pas une tuile donne un message lisible, sans le token."""
    body = b"<html>Acces bloque, token=secret123</html>"
    response = SimpleNamespace(status_code=200, content=body,
                               headers={"Content-Type": "text/html"})
    monkeypatch.setattr(mc, "_get", lambda *a, **k: response)
    with pytest.raises(RuntimeError) as error:
        mc.list_candidate_images(LAT, LON, 200, token="secret123", source="tiles")
    message = str(error.value)
    assert "text/html" in message and "Acces bloque" in message
    assert "secret123" not in message



# ---------------------------------------------------------------------------
# Liste par l'API Graph (source par défaut)
# ---------------------------------------------------------------------------
class FakeJsonResponse:
    """Réponse simulée de l'API Graph."""

    def __init__(self, payload=None, status_code=200, body=None, content_type="application/json"):
        self.payload, self.status_code = payload, status_code
        self.content = body if body is not None else b"{}"
        self.headers = {"Content-Type": content_type}

    def json(self):
        if self.payload is None:
            raise ValueError("pas du JSON")
        return self.payload


def graph_item(image_id, north_m, seq="s1"):
    return {"id": str(image_id), "geometry": {"type": "Point",
                                              "coordinates": [LON, LAT + north_m * DEG_PER_M]},
            "sequence": seq, "captured_at": 1_700_000_000_000, "compass_angle": 10.0,
            "is_pano": False, "thumb_2048_url": f"https://exemple/{image_id}.jpg"}


def test_graph_filtre_le_cercle_et_garde_la_vignette(monkeypatch):
    payload = {"data": [graph_item(1, 50), graph_item(2, 150), graph_item(3, 260)]}
    monkeypatch.setattr(mc, "_get", lambda *a, **k: FakeJsonResponse(payload))
    images = mc.list_candidate_images(LAT, LON, 200, token="t")
    assert [img["image_id"] for img in images] == ["1", "2"]
    assert images[0]["thumb_2048_url"] == "https://exemple/1.jpg"
    assert images[0]["sequence_id"] == "s1"


def test_graph_decoupe_la_boite_si_limite_atteinte(monkeypatch):
    monkeypatch.setattr(config, "MAPILLARY_GRAPH_LIMIT", 2)
    calls = []

    def fake_get(url, session, **kwargs):
        calls.append(url)
        if len(calls) == 1:                      # première boîte : limite atteinte
            return FakeJsonResponse({"data": [graph_item(1, 10), graph_item(2, 20)]})
        return FakeJsonResponse({"data": [graph_item(len(calls), 10)]})

    monkeypatch.setattr(mc, "_get", fake_get)
    items = mc._graph_images_in_bbox(mc.bbox_from_radius(LAT, LON, 200), "t", None)
    assert len(calls) == 5                       # la boîte puis ses quatre quarts
    assert {item["id"] for item in items} == {"2", "3", "4", "5"}


def test_grille_couvre_la_boite():
    bbox = mc.bbox_from_radius(LAT, LON, 200)
    cells = mc._grid(bbox, 50)
    assert len(cells) == 64                      # 400 m de côté / 50 m = 8 x 8 cases
    assert cells[0][:2] == pytest.approx(bbox[:2])
    assert cells[-1][2:] == pytest.approx(bbox[2:])
    west, south, east, north = cells[0]
    assert mc.haversine_m(south, west, south, east) <= 50.5
    assert mc.haversine_m(south, west, north, west) <= 50.5


def test_passes_fusionnent_jusqu_a_stabilite(monkeypatch):
    monkeypatch.setattr(config, "MAPILLARY_CELL_M", 10_000)   # une seule case
    answers = iter([[graph_item(1, 10), graph_item(2, 20)],   # passe 1
                    [graph_item(2, 20), graph_item(3, 30)],   # passe 2 : une nouvelle
                    [graph_item(3, 30)],                      # passe 3 : rien de neuf
                    [graph_item(4, 40)]])                     # jamais demandée
    monkeypatch.setattr(mc, "_graph_images_in_bbox", lambda *a, **k: next(answers))
    images = mc.list_candidate_images(LAT, LON, 200, token="t")
    assert {img["image_id"] for img in images} == {"1", "2", "3"}


def test_case_en_echec_ignoree(monkeypatch):
    monkeypatch.setattr(config, "MAPILLARY_CELL_M", 10_000)
    calls = []

    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise mc.TransientError("échec après 3 tentatives")
        return [graph_item(1, 10)]

    monkeypatch.setattr(mc, "_graph_images_in_bbox", fake)
    images = mc.list_candidate_images(LAT, LON, 200, token="t")
    assert [img["image_id"] for img in images] == ["1"]


def test_collect_reutilise_la_liste_existante(tmp_path, monkeypatch):
    candidates = [make_image(str(i), f"s{i}", i, 1_700_000_000_000 + i) for i in range(10)]
    monkeypatch.setenv(config.MAPILLARY_TOKEN_ENV, "token-de-test")
    monkeypatch.setattr(mc, "fetch_image_info",
                        lambda i, t, session=None: {"thumb_2048_url": "u", "is_pano": False})
    monkeypatch.setattr(mc, "download_image", lambda url, path, session=None: True)

    def selected(path):
        with open(path, encoding="utf-8") as handle:
            return {r["image_id"] for r in csv.DictReader(handle) if r["selected"] == "True"}

    monkeypatch.setattr(mc, "list_candidate_images", lambda *a, **k: candidates)
    first = selected(mc.collect(LAT, LON, 200, tmp_path, spacing_m=0, max_images=4))

    # seconde exécution : l'API n'est plus interrogée, la sélection est identique
    monkeypatch.setattr(mc, "list_candidate_images",
                        lambda *a, **k: pytest.fail("la liste aurait dû être reprise"))
    assert selected(mc.collect(LAT, LON, 200, tmp_path, spacing_m=0, max_images=4)) == first

    # avec refresh, l'API est de nouveau interrogée
    asked = []
    monkeypatch.setattr(mc, "list_candidate_images",
                        lambda *a, **k: asked.append(1) or candidates)
    mc.collect(LAT, LON, 200, tmp_path, spacing_m=0, max_images=4, refresh=True)
    assert asked == [1]


def test_graph_erreur_json_lisible(monkeypatch):
    payload = {"error": {"message": "Invalid OAuth access token.", "code": 190}}
    monkeypatch.setattr(mc, "_get", lambda *a, **k: FakeJsonResponse(payload, status_code=400))
    with pytest.raises(RuntimeError, match="Invalid OAuth access token"):
        mc.list_candidate_images(LAT, LON, 200, token="t")


def test_graph_page_html_lisible(monkeypatch):
    response = FakeJsonResponse(None, body=b"<!DOCTYPE html><html id='facebook'>secret</html>",
                                content_type="text/html")
    monkeypatch.setattr(mc, "_get", lambda *a, **k: response)
    with pytest.raises(RuntimeError) as error:
        mc.list_candidate_images(LAT, LON, 200, token="secret")
    assert "text/html" in str(error.value) and "secret" not in str(error.value)


def test_collect_reutilise_la_vignette_de_l_api_graph(tmp_path, monkeypatch):
    candidates = [dict(make_image("1", "s1", 0, 1_700_000_000_000),
                       thumb_2048_url="https://exemple/1.jpg")]
    monkeypatch.setenv(config.MAPILLARY_TOKEN_ENV, "token-de-test")
    monkeypatch.setattr(mc, "list_candidate_images", lambda *a, **k: candidates)
    monkeypatch.setattr(mc, "fetch_image_info",
                        lambda *a, **k: pytest.fail("pas de seconde requête attendue"))
    urls = []
    monkeypatch.setattr(mc, "download_image",
                        lambda url, path, session=None: urls.append(url) or True)
    mc.collect(LAT, LON, 200, tmp_path, spacing_m=15)
    assert urls == ["https://exemple/1.jpg"]

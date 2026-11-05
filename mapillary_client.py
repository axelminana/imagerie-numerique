"""Étape 1 : collecte des images Mapillary autour d'un point (responsable : R1).

Produit, dans le dossier de la zone :
    - images/<image_id>.jpg : vignettes 2048 px des images retenues
    - metadata.csv          : toutes les images candidates (colonnes : config.METADATA_COLUMNS)

Déroulement (section 6.1 du plan de travail) :
    1. boîte englobante calculée à partir du rayon ;
    2. liste des images de la boîte avec l'API Graph (ou les tuiles vectorielles),
       puis filtrage par distance (haversine) ;
    3. exclusion des panoramas, sous-échantillonnage par séquence (--spacing) ;
    4. limitation éventuelle du nombre d'images (--max-images, tirage reproductible) ;
    5. téléchargement des vignettes ;
    6. écriture de metadata.csv.

Utilisable seul pour compter les images d'une zone avant de la choisir :
    python mapillary_client.py -lat 45.800413 -lon 3.108813 --radius 200
"""
from __future__ import annotations

import argparse
import csv
import gzip
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

import config

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000          # rayon moyen de la Terre en mètres
METERS_PER_DEGREE_LAT = 111_320     # longueur approximative d'un degré de latitude
RETRY_STATUS = {429, 500, 502, 503, 504}   # codes HTTP pour lesquels on réessaie


# ---------------------------------------------------------------------------
# Fonctions géométriques
# ---------------------------------------------------------------------------
def bbox_from_radius(lat: float, lon: float,
                     radius_m: float) -> tuple[float, float, float, float]:
    """Retourne (ouest, sud, est, nord) de la boîte qui contient le cercle de rayon radius_m."""
    dlat = radius_m / METERS_PER_DEGREE_LAT
    dlon = radius_m / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en mètres entre deux points GPS (formule de haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Accès réseau
# ---------------------------------------------------------------------------
def get_token() -> str:
    """Lit le token Mapillary dans la variable d'environnement MAPILLARY_TOKEN.

    Le fichier .env est chargé s'il existe (bibliothèque python-dotenv).
    Lève une erreur explicite si le token est absent : il ne doit jamais
    être écrit dans le code.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    token = os.environ.get(config.MAPILLARY_TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"Token Mapillary introuvable : définissez la variable {config.MAPILLARY_TOKEN_ENV} "
            "(voir .env.example)."
        )
    return token


def _get(url: str, session: requests.Session, headers: dict | None = None,
         stream: bool = False) -> requests.Response:
    """Requête GET avec délai maximal et nouvelles tentatives.

    On réessaie après une erreur réseau ou un code HTTP d'erreur temporaire
    (RETRY_STATUS), en attendant 1 s, 2 s, 4 s... Les autres codes (401, 404...)
    sont renvoyés tels quels à l'appelant, qui décide quoi en faire.
    L'URL n'est jamais journalisée car celle des tuiles contient le token.
    """
    last_error: Exception | None = None
    for attempt in range(config.REQUEST_RETRIES):
        try:
            response = session.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT_S,
                                   stream=stream)
            if response.status_code not in RETRY_STATUS:
                return response
            last_error = RuntimeError(f"code HTTP {response.status_code}")
        except (requests.ConnectionError, requests.Timeout) as error:
            last_error = error
        if attempt < config.REQUEST_RETRIES - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"échec après {config.REQUEST_RETRIES} tentatives : {last_error}")


def _new_session() -> requests.Session:
    """Session HTTP avec un User-Agent explicite (certains serveurs de Meta
    répondent par une page web au client par défaut "python-requests")."""
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    return session


def _check_auth(response: requests.Response) -> None:
    """Arrête le programme avec un message clair si le token est refusé."""
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Mapillary refuse le token (code HTTP {response.status_code}) : "
            f"vérifiez la valeur de {config.MAPILLARY_TOKEN_ENV} dans .env."
        )


def _describe(response: requests.Response, token: str) -> str:
    """Résumé lisible d'une réponse inattendue (token masqué), pour les messages d'erreur."""
    preview = response.content[:300].decode("utf-8", errors="replace").replace(token, "<token>")
    return (f"code HTTP {response.status_code}, "
            f"type {response.headers.get('Content-Type', 'inconnu')}. "
            f"Début de la réponse : {preview!r}")


def _decode_tile(content: bytes, tile) -> list[dict]:
    """Décode une tuile vectorielle et retourne les entités de la couche "image"."""
    from vt2geojson.tools import vt_bytes_to_geojson
    # certaines tuiles arrivent compressées en gzip sans que requests le sache
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    geojson = vt_bytes_to_geojson(content, tile.x, tile.y, tile.z,
                                  layer=config.MAPILLARY_TILE_LAYER)
    return geojson.get("features", [])


def _ms_to_iso(value) -> str:
    """Convertit une date Mapillary (millisecondes depuis 1970) en texte ISO 8601 (UTC)."""
    if value in (None, ""):
        return ""
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return str(value)


# ---------------------------------------------------------------------------
# Deux sources pour lister les images d'une boîte
# ---------------------------------------------------------------------------
def _graph_json(response: requests.Response, token: str) -> dict:
    """Contenu JSON d'une réponse de l'API Graph, ou erreur explicite."""
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            f"L'API Graph n'a pas renvoyé de JSON ({_describe(response, token)})"
        ) from None
    if isinstance(data, dict) and "error" in data:
        error = data["error"]
        message = error.get("message", error) if isinstance(error, dict) else error
        raise RuntimeError(
            f"L'API Graph refuse la requête (code HTTP {response.status_code}) : {message}. "
            f"Vérifiez la valeur de {config.MAPILLARY_TOKEN_ENV} dans .env."
        )
    if response.status_code != 200:
        raise RuntimeError(f"Réponse inattendue de l'API Graph ({_describe(response, token)})")
    return data


def _graph_images_in_bbox(bbox: tuple[float, float, float, float], token: str,
                          session: requests.Session, depth: int = 0) -> list[dict]:
    """Images d'une boîte via le point d'accès /images de l'API Graph.

    L'API renvoie au plus config.MAPILLARY_GRAPH_LIMIT images par requête : si
    la limite est atteinte, la boîte est découpée en quatre et chaque quart est
    interrogé séparément.
    """
    west, south, east, north = bbox
    url = (f"{config.MAPILLARY_GRAPH_IMAGES_URL}?access_token={token}"
           f"&fields={config.MAPILLARY_LIST_FIELDS}"
           f"&bbox={west:.6f},{south:.6f},{east:.6f},{north:.6f}"
           f"&limit={config.MAPILLARY_GRAPH_LIMIT}")
    items = _graph_json(_get(url, session), token).get("data", [])

    if len(items) >= config.MAPILLARY_GRAPH_LIMIT:
        if depth < config.MAPILLARY_MAX_SPLIT_DEPTH:
            mid_lon, mid_lat = (west + east) / 2, (south + north) / 2
            quarters = [(west, south, mid_lon, mid_lat), (mid_lon, south, east, mid_lat),
                        (west, mid_lat, mid_lon, north), (mid_lon, mid_lat, east, north)]
            log.debug("Limite de %d images atteinte : découpage de la boîte (niveau %d)",
                      config.MAPILLARY_GRAPH_LIMIT, depth + 1)
            return [item for quarter in quarters
                    for item in _graph_images_in_bbox(quarter, token, session, depth + 1)]
        log.warning("Limite de %d images atteinte sur une petite boîte : "
                    "quelques images peuvent manquer", config.MAPILLARY_GRAPH_LIMIT)
    return items


def _records_from_graph(bbox, token: str, session: requests.Session):
    """Images de la boîte, lues avec l'API Graph (source par défaut)."""
    for item in _graph_images_in_bbox(bbox, token, session):
        coordinates = (item.get("geometry") or {}).get("coordinates")
        if not coordinates:
            continue
        sequence = item.get("sequence")
        if isinstance(sequence, dict):          # selon les versions, objet ou identifiant
            sequence = sequence.get("id")
        yield {
            "id": item.get("id"), "lon": coordinates[0], "lat": coordinates[1],
            "sequence_id": sequence, "captured_at": item.get("captured_at"),
            "compass_angle": item.get("compass_angle"), "is_pano": item.get("is_pano"),
            "thumb_2048_url": item.get("thumb_2048_url"),
        }


def _records_from_tiles(bbox, token: str, session: requests.Session):
    """Images de la boîte, lues dans les tuiles vectorielles (méthode de test_student.py)."""
    import mercantile
    from tqdm import tqdm

    west, south, east, north = bbox
    tiles = list(mercantile.tiles(west, south, east, north, config.MAPILLARY_TILE_ZOOM))
    log.info("%d tuile(s) à lire", len(tiles))
    for tile in tqdm(tiles, desc="Tuiles", unit="tuile"):
        url = config.MAPILLARY_TILE_URL.format(z=tile.z, x=tile.x, y=tile.y, token=token)
        try:
            response = _get(url, session)
        except RuntimeError as error:
            log.warning("Tuile %s/%s/%s ignorée : %s", tile.z, tile.x, tile.y, error)
            continue
        _check_auth(response)
        if response.status_code != 200 or not response.content:
            log.warning("Tuile %s/%s/%s ignorée (code HTTP %s)",
                        tile.z, tile.x, tile.y, response.status_code)
            continue
        try:
            features = _decode_tile(response.content, tile)
        except Exception as error:  # DecodeError de protobuf, gzip invalide...
            raise RuntimeError(
                f"La tuile {tile.z}/{tile.x}/{tile.y} n'est pas une tuile vectorielle valide "
                f"({_describe(response, token)})"
            ) from error
        for feature in features:
            props = feature.get("properties", {})
            img_lon, img_lat = feature["geometry"]["coordinates"][:2]
            yield {
                "id": props.get("id"), "lon": img_lon, "lat": img_lat,
                "sequence_id": props.get("sequence_id"), "captured_at": props.get("captured_at"),
                "compass_angle": props.get("compass_angle"), "is_pano": props.get("is_pano"),
                "thumb_2048_url": None,
            }


# ---------------------------------------------------------------------------
# Étapes de la collecte
# ---------------------------------------------------------------------------
def list_candidate_images(lat: float, lon: float, radius_m: float, token: str,
                          session: requests.Session | None = None,
                          source: str = config.MAPILLARY_SOURCE) -> list[dict]:
    """Liste les images Mapillary situées à moins de radius_m mètres du point.

    source vaut "graph" (API Graph, une requête pour toute la zone, par défaut)
    ou "tiles" (tuiles vectorielles, méthode de test_student.py). Dans les deux
    cas, la boîte englobante déborde du cercle : on ne garde que les images à
    moins de radius_m mètres (haversine_m).

    Retourne une liste de dictionnaires triée par distance, avec les clés :
    image_id, sequence_id, lat, lon, distance_m, captured_at_ms, compass_angle,
    is_pano et thumb_2048_url (None si la source ne le fournit pas).
    """
    if source not in ("graph", "tiles"):
        raise ValueError(f"source inconnue : {source!r} (choix : graph, tiles)")
    session = session or _new_session()
    bbox = bbox_from_radius(lat, lon, radius_m)
    records = (_records_from_graph if source == "graph" else _records_from_tiles)(
        bbox, token, session)

    images: dict[str, dict] = {}
    for record in records:
        image_id = str(record.get("id") or "")
        if not image_id or image_id in images:
            continue
        distance = haversine_m(lat, lon, record["lat"], record["lon"])
        if distance > radius_m:
            continue
        images[image_id] = {
            "image_id": image_id,
            "sequence_id": str(record.get("sequence_id") or ""),
            "lat": record["lat"],
            "lon": record["lon"],
            "distance_m": round(distance, 1),
            "captured_at_ms": record.get("captured_at"),
            "compass_angle": record.get("compass_angle"),
            "is_pano": record.get("is_pano"),
            "thumb_2048_url": record.get("thumb_2048_url"),
        }

    log.info("%d image(s) dans un rayon de %s m (source : %s)", len(images), radius_m, source)
    return sorted(images.values(), key=lambda img: img["distance_m"])


def subsample_by_spacing(images: list[dict], spacing_m: float) -> list[dict]:
    """Garde, dans chaque séquence, une image tous les spacing_m mètres au minimum.

    Les images d'une séquence sont triées par date ; une image est gardée si elle
    est à au moins spacing_m mètres de la dernière image gardée de sa séquence.
    Sans ce tri, un même véhicule serait compté sur plusieurs images successives.
    Une valeur de spacing_m nulle ou négative garde toutes les images.
    """
    if spacing_m <= 0:
        return list(images)

    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for img in images:
        by_sequence[img["sequence_id"]].append(img)

    kept: list[dict] = []
    for sequence in by_sequence.values():
        # les images sans date passent en dernier, triées par identifiant
        sequence.sort(key=lambda img: (img.get("captured_at_ms") is None,
                                       img.get("captured_at_ms") or 0, img["image_id"]))
        last = None
        for img in sequence:
            if last is None or haversine_m(last["lat"], last["lon"],
                                           img["lat"], img["lon"]) >= spacing_m:
                kept.append(img)
                last = img
    return kept


def fetch_image_info(image_id: str, token: str,
                     session: requests.Session | None = None) -> dict | None:
    """Interroge l'API Graph pour une image (champs : config.MAPILLARY_IMAGE_FIELDS).

    Retourne le dictionnaire JSON renvoyé par l'API, ou None (après journalisation)
    si l'image est inaccessible ou si thumb_2048_url est absent.
    """
    session = session or _new_session()
    url = (config.MAPILLARY_GRAPH_URL.format(image_id=image_id)
           + f"?fields={config.MAPILLARY_IMAGE_FIELDS}")
    try:
        response = _get(url, session, headers={"Authorization": f"OAuth {token}"})
    except RuntimeError as error:
        log.warning("Image %s : métadonnées indisponibles (%s)", image_id, error)
        return None
    _check_auth(response)
    if response.status_code != 200:
        log.warning("Image %s : code HTTP %s", image_id, response.status_code)
        return None
    try:
        info = response.json()
    except ValueError:
        log.warning("Image %s : réponse illisible", image_id)
        return None
    if not info.get("thumb_2048_url"):
        log.warning("Image %s : pas de vignette 2048 px", image_id)
        return None
    return info


def download_image(url: str, path: Path, session: requests.Session | None = None) -> bool:
    """Télécharge une vignette vers path, sauf si le fichier existe déjà.

    Le fichier est d'abord écrit sous un nom temporaire puis renommé : une
    interruption ne laisse jamais d'image à moitié écrite.
    Retourne True si le fichier est présent à la fin de l'appel.
    """
    if path.exists() and path.stat().st_size > 0:
        return True
    session = session or _new_session()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    try:
        response = _get(url, session, stream=True)
        if response.status_code != 200:
            log.warning("%s : téléchargement refusé (code HTTP %s)", path.name, response.status_code)
            return False
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                handle.write(chunk)
        tmp_path.replace(path)
        return True
    except (RuntimeError, requests.RequestException, OSError) as error:
        log.warning("%s : téléchargement impossible (%s)", path.name, error)
        tmp_path.unlink(missing_ok=True)
        return False


def write_metadata(rows: list[dict], path: Path) -> None:
    """Écrit metadata.csv avec exactement les colonnes de config.METADATA_COLUMNS."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.METADATA_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_images(lat: float, lon: float, radius_m: float = config.DEFAULT_RADIUS_M,
                 spacing_m: float = config.DEFAULT_SPACING_M,
                 source: str = config.MAPILLARY_SOURCE) -> tuple[int, int]:
    """Compte les images disponibles dans le rayon, sans rien télécharger.

    Retourne (nombre d'images dans le rayon, nombre restant après exclusion des
    panoramas connus et sous-échantillonnage). Sert à vérifier la densité
    d'images avant de choisir une zone (section 7 du plan).
    """
    candidates = list_candidate_images(lat, lon, radius_m, get_token(), source=source)
    usable = [img for img in candidates if not img.get("is_pano")]
    return len(candidates), len(subsample_by_spacing(usable, spacing_m))


def collect(lat: float, lon: float, radius_m: float, zone_dir: Path,
            spacing_m: float = config.DEFAULT_SPACING_M,
            max_images: int | None = None, seed: int = config.RANDOM_SEED) -> Path:
    """Exécute toute l'étape 1 et retourne le chemin de metadata.csv.

    metadata.csv contient toutes les images candidates du cercle ; la colonne
    selected indique celles qui ont été retenues et téléchargées, et path leur
    chemin relatif au dossier de la zone (images/<image_id>.jpg).
    """
    from tqdm import tqdm

    token = get_token()
    session = _new_session()
    zone_dir = Path(zone_dir)
    images_dir = zone_dir / config.IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. images candidates dans le cercle
    candidates = list_candidate_images(lat, lon, radius_m, token, session)
    if not candidates:
        log.warning("Aucune image Mapillary dans un rayon de %s m : zone à changer ?", radius_m)

    # 2. exclusion des panoramas connus, puis sous-échantillonnage par séquence
    usable = [img for img in candidates if not img.get("is_pano")]
    kept = subsample_by_spacing(usable, spacing_m)
    log.info("%d panorama(s) exclu(s), %d image(s) après sous-échantillonnage (%s m)",
             len(candidates) - len(usable), len(kept), spacing_m)

    # 3. limitation éventuelle, par tirage reproductible
    if max_images is not None and len(kept) > max_images:
        kept = random.Random(seed).sample(kept, max_images)
        log.info("Tirage de %d image(s) (graine %s)", max_images, seed)
    kept_ids = {img["image_id"] for img in kept}

    # 4. métadonnées et téléchargement des images retenues
    rows: dict[str, dict] = {}
    for img in candidates:
        rows[img["image_id"]] = {
            **img,
            "captured_at": _ms_to_iso(img.get("captured_at_ms")),
            "selected": False,
            "path": "",
        }

    n_ok = 0
    for image_id in tqdm(sorted(kept_ids), desc="Images", unit="img"):
        row = rows[image_id]
        if row.get("thumb_2048_url"):
            # l'API Graph a déjà fourni la vignette : pas de seconde requête
            info = {"thumb_2048_url": row["thumb_2048_url"], "is_pano": row.get("is_pano"),
                    "captured_at": row.get("captured_at_ms"),
                    "compass_angle": row.get("compass_angle")}
        else:
            info = fetch_image_info(image_id, token, session)
        if info is None:
            continue
        # l'API Graph complète ou corrige ce que la tuile ne donnait pas
        if info.get("is_pano"):
            row["is_pano"] = True
            continue
        row["is_pano"] = bool(info.get("is_pano", row.get("is_pano") or False))
        if info.get("captured_at") is not None:
            row["captured_at"] = _ms_to_iso(info["captured_at"])
        if info.get("compass_angle") is not None:
            row["compass_angle"] = info["compass_angle"]

        relative = Path(config.IMAGES_DIR) / f"{image_id}.jpg"
        if download_image(info["thumb_2048_url"], zone_dir / relative, session):
            row["selected"] = True
            row["path"] = relative.as_posix()
            n_ok += 1

    # 5. écriture de metadata.csv (les plus proches du centre en premier)
    metadata_path = zone_dir / config.METADATA_FILE
    ordered = sorted(rows.values(), key=lambda r: r["distance_m"])
    for row in ordered:
        if row.get("is_pano") is None:
            row["is_pano"] = ""
    write_metadata(ordered, metadata_path)
    log.info("%d image(s) téléchargée(s) sur %d retenue(s) ; métadonnées dans %s",
             n_ok, len(kept_ids), metadata_path)
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    """Compte les images disponibles autour d'un point (aide au choix des zones)."""
    parser = argparse.ArgumentParser(
        description="Compte les images Mapillary disponibles autour d'un point.")
    parser.add_argument("-lat", type=float, required=True, help="latitude du centre")
    parser.add_argument("-lon", type=float, required=True, help="longitude du centre")
    parser.add_argument("--radius", type=float, default=config.DEFAULT_RADIUS_M,
                        help="rayon en mètres (défaut : %(default)s)")
    parser.add_argument("--spacing", type=float, default=config.DEFAULT_SPACING_M,
                        help="espacement du sous-échantillonnage en mètres (défaut : %(default)s)")
    parser.add_argument("--source", choices=("graph", "tiles"), default=config.MAPILLARY_SOURCE,
                        help="API Graph ou tuiles vectorielles (défaut : %(default)s)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")

    try:
        total, usable = count_images(args.lat, args.lon, args.radius, args.spacing, args.source)
    except RuntimeError as error:
        log.error("%s", error)
        return 1
    print(f"{total} image(s) dans un rayon de {args.radius:g} m, "
          f"{usable} après exclusion des panoramas et sous-échantillonnage ({args.spacing:g} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

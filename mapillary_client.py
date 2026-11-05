"""Étape 1 : collecte des images Mapillary autour d'un point (responsable : R1).

Produit, dans le dossier de la zone :
    - images/<image_id>.jpg : vignettes 2048 px des images retenues
    - metadata.csv          : toutes les images candidates (colonnes : config.METADATA_COLUMNS)

Voir la section 6.1 du plan de travail et l'annexe A pour les défauts de
test_student.py à ne pas reproduire.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import config

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000          # rayon moyen de la Terre en mètres
METERS_PER_DEGREE_LAT = 111_320     # longueur approximative d'un degré de latitude


# ---------------------------------------------------------------------------
# Fonctions géométriques (déjà écrites et testées dans tests/test_geo.py)
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


# ---------------------------------------------------------------------------
# À implémenter (R1)
# ---------------------------------------------------------------------------
def list_candidate_images(lat: float, lon: float, radius_m: float, token: str) -> list[dict]:
    """Liste les images Mapillary situées à moins de radius_m mètres du point.

    Étapes : boîte englobante (bbox_from_radius), tuiles au zoom 14 (mercantile),
    lecture de la couche "image" (vt2geojson), filtrage par distance (haversine_m).

    Retourne une liste de dictionnaires avec au moins les clés :
    image_id, sequence_id, lat, lon, distance_m, captured_at.
    """
    raise NotImplementedError("TODO (R1) : lister les images candidates")


def subsample_by_spacing(images: list[dict], spacing_m: float) -> list[dict]:
    """Garde, dans chaque séquence, une image tous les spacing_m mètres au minimum.

    Les images d'une séquence sont triées par date ; une image est gardée si elle
    est à au moins spacing_m mètres de la dernière image gardée de sa séquence.
    """
    raise NotImplementedError("TODO (R1) : sous-échantillonnage par séquence")


def fetch_image_info(image_id: str, token: str) -> dict | None:
    """Interroge l'API Graph pour une image (champs : config.MAPILLARY_IMAGE_FIELDS).

    Utilise un délai maximal (config.REQUEST_TIMEOUT_S) et plusieurs tentatives
    (config.REQUEST_RETRIES). Retourne None, après journalisation, si l'image
    est inaccessible ou si thumb_2048_url est absent.
    """
    raise NotImplementedError("TODO (R1) : métadonnées d'une image")


def download_image(url: str, path: Path) -> bool:
    """Télécharge une vignette vers path, sauf si le fichier existe déjà.

    Retourne True si le fichier est présent à la fin de l'appel.
    """
    raise NotImplementedError("TODO (R1) : téléchargement d'une image")


def count_images(lat: float, lon: float, radius_m: float = config.DEFAULT_RADIUS_M) -> int:
    """Compte les images disponibles dans le rayon, sans rien télécharger.

    Sert à vérifier la densité d'images avant de choisir une zone (section 7).
    """
    return len(list_candidate_images(lat, lon, radius_m, get_token()))


def collect(lat: float, lon: float, radius_m: float, zone_dir: Path,
            spacing_m: float = config.DEFAULT_SPACING_M,
            max_images: int | None = None) -> Path:
    """Exécute toute l'étape 1 et retourne le chemin de metadata.csv.

    1. lister les candidates ; 2. sous-échantillonner ; 3. limiter à max_images
    (tirage avec config.RANDOM_SEED) ; 4. récupérer les métadonnées, exclure les
    panoramas ; 5. télécharger dans zone_dir/images/ ; 6. écrire metadata.csv
    avec toutes les candidates (colonne selected).
    """
    raise NotImplementedError("TODO (R1) : enchaîner la collecte complète")

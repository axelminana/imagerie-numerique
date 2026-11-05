"""Paramètres communs à tous les modules du projet.

Ce fichier centralise les catégories, la correspondance avec les classes COCO,
les noms et colonnes des fichiers intermédiaires (section 5 du plan de travail)
et les valeurs par défaut. Toute modification des contrats de données se fait
ici et doit être annoncée à tout le groupe.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Catégories du projet (ordre des colonnes de report.csv)
# ---------------------------------------------------------------------------
CATEGORIES: list[str] = [
    "pietons",
    "deux_roues",
    "deux_roues_motorises",
    "vehicules_legers",
    "bus",
    "camions",
]

# Correspondance classe COCO renvoyée par YOLO -> catégorie du projet.
# Les classes absentes de ce dictionnaire sont ignorées.
COCO_TO_CATEGORY: dict[str, str] = {
    "person": "pietons",
    "bicycle": "deux_roues",
    "motorcycle": "deux_roues_motorises",
    "car": "vehicules_legers",
    "bus": "bus",
    "truck": "camions",
}

# Catégories concernées par le tri stationné / en circulation (section 6.4).
VEHICLE_CATEGORIES: list[str] = [c for c in CATEGORIES if c != "pietons"]

# Catégories sur lesquelles une personne peut être conductrice (section 6.3).
TWO_WHEELER_CATEGORIES: tuple[str, ...] = ("deux_roues", "deux_roues_motorises")

# ---------------------------------------------------------------------------
# Dossiers et fichiers
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
IMAGES_DIR = "images"
FIGURES_DIR = "figures"

METADATA_FILE = "metadata.csv"
FILTER_FILE = "filter.csv"
DETECTIONS_FILE = "detections.csv"
REPORT_FILE = "report.csv"
STATS_FILE = "stats.csv"

# ---------------------------------------------------------------------------
# Contrats de données : colonnes de chaque fichier CSV (section 5 du plan)
# ---------------------------------------------------------------------------
METADATA_COLUMNS: list[str] = [
    "image_id", "sequence_id", "lat", "lon", "distance_m",
    "captured_at", "compass_angle", "is_pano", "selected", "path",
]
FILTER_COLUMNS: list[str] = ["image_id", "method", "answer", "score", "keep"]
DETECTION_COLUMNS: list[str] = [
    "image_id", "coco_class", "category", "conf",
    "x1", "y1", "x2", "y2", "is_rider", "parked", "parked_method",
]
REPORT_COLUMNS: list[str] = ["image"] + CATEGORIES
STATS_COLUMNS: list[str] = [
    "zone", "category", "count", "share", "share_low", "share_high",
    "presence_rate", "mean_per_image",
]
ANNOTATION_COLUMNS: list[str] = (
    ["image_id", "split", "keep"] + CATEGORIES + ["stationnes", "annotateur", "commentaire"]
)

# ---------------------------------------------------------------------------
# Étape 1 : collecte Mapillary
# ---------------------------------------------------------------------------
MAPILLARY_TOKEN_ENV = "MAPILLARY_TOKEN"     # nom de la variable d'environnement
MAPILLARY_TILE_URL = (
    "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={token}"
)
MAPILLARY_GRAPH_URL = "https://graph.mapillary.com/{image_id}"
MAPILLARY_IMAGE_FIELDS = "thumb_2048_url,captured_at,compass_angle,is_pano"
MAPILLARY_GRAPH_IMAGES_URL = "https://graph.mapillary.com/images"
MAPILLARY_LIST_FIELDS = "id,geometry,sequence,captured_at,compass_angle,is_pano,thumb_2048_url"
MAPILLARY_GRAPH_LIMIT = 2000                # maximum d'images par requête /images
MAPILLARY_MAX_SPLIT_DEPTH = 4               # découpages successifs de la boîte au maximum
MAPILLARY_SOURCE = "graph"                  # "graph" ou "tiles" (méthode de test_student.py)
USER_AGENT = "projet-imagerie-numerique/1.0 (Polytech Clermont)"
MAPILLARY_TILE_ZOOM = 14                    # seul niveau de zoom servi par l'API
MAPILLARY_TILE_LAYER = "image"

DEFAULT_RADIUS_M = 200
DEFAULT_SPACING_M = 15                      # à valider le 12/11
REQUEST_TIMEOUT_S = 20
REQUEST_RETRIES = 3

# ---------------------------------------------------------------------------
# Étape 2 : filtrage vision-langage (valeurs à régler sur la partie réglage)
# ---------------------------------------------------------------------------
BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
BLIP2_PROMPTS: list[str] = [
    "Question: Is this a street scene viewed from a road? Answer yes or no. Answer:",
    "Question: Does this image show a road open to traffic? Answer yes or no. Answer:",
    "Question: Was this photo taken from a street or a sidewalk? Answer yes or no. Answer:",
]
BLIP2_REJECT_PROMPT = "Question: Is this a parking lot? Answer yes or no. Answer:"

CLIP_MODEL = "openai/clip-vit-large-patch14"
CLIP_POSITIVE_PROMPTS: list[str] = [
    "a photo of a street with traffic taken from the road",
    "a city street seen from the sidewalk",
    "a road seen from a car dashboard camera",
]
CLIP_NEGATIVE_PROMPTS: list[str] = [
    "a parking lot full of parked cars",
    "the inside of a car",
    "a building facade",
    "a blurry photo",
]
CLIP_KEEP_THRESHOLD = 0.5                   # à régler

# ---------------------------------------------------------------------------
# Étape 3 : détection YOLO
# ---------------------------------------------------------------------------
DEFAULT_DETECTOR = "yolo11s.pt"
DEFAULT_CONF = 0.4                          # à régler entre 0,25 et 0,5
DEFAULT_IMGSZ = 1280
RIDER_OVERLAP_THRESHOLD = 0.3               # part de la boîte de la personne couverte
PARKED_STRATEGIES: tuple[str, ...] = ("none", "geometry", "clip")
DEFAULT_PARKED_STRATEGY = "geometry"        # à choisir le 10/12

# ---------------------------------------------------------------------------
# Étape 5 : analyse
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

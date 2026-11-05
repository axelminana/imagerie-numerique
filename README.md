# Estimation des modalités de mobilité à partir d'images

Projet de groupe, Polycompétence Imagerie Numérique (Université Clermont Auvergne, 2026-2027).

À partir d'une latitude et d'une longitude, l'application télécharge les images
Mapillary prises dans un rayon d'environ 200 m, écarte les images inutilisables
avec un modèle vision-langage, détecte les usagers de la route avec YOLO et
produit un fichier `report.csv` avec le nombre d'objets de chaque catégorie par
image. Une analyse statistique compare ensuite trois zones géographiques.

Le plan de travail complet est dans [`docs/`](docs/).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt    # installer torch selon la version CUDA du GPU
cp .env.example .env               # puis y mettre son propre token Mapillary
```

## Utilisation

```bash
# pipeline complet sur une zone
python main.py -lat 45.800413 -lon 3.108813 -o report.csv --zone carrefour_pistes

# essai rapide sur CPU avec le plus petit modèle YOLO
python main.py -lat 45.800413 -lon 3.108813 --zone carrefour_pistes --max-images 20 \
               --detector yolo11n.pt --device cpu

# relancer seulement certaines étapes
python main.py -lat 45.800413 -lon 3.108813 --zone carrefour_pistes --steps detect,report

# analyse comparée de plusieurs zones
python analysis.py --zones data/zone_a data/zone_b data/zone_c --out figures/

# évaluation sur le jeu annoté
python evaluate.py --zone data/carrefour_pistes --split test

# compter les images disponibles autour d'un point (choix des zones)
python mapillary_client.py -lat 45.800413 -lon 3.108813 --radius 200

# aide complète
python main.py -h
```

## Organisation

| Fichier | Étape | Rôle |
|---|---|---|
| `main.py` | toutes | Options de la ligne de commande et enchaînement des étapes | 
| `config.py` | toutes | Catégories, correspondance COCO, colonnes des CSV, valeurs par défaut |
| `mapillary_client.py` | 1. Collecte | Images dans le rayon, sous-échantillonnage, téléchargement |
| `filtering.py` | 2. Filtrage | BLIP-2 et CLIP |
| `detection.py` | 3. Détection | YOLO, regroupement des classes, dédoublonnage des conducteurs |
| `parked.py` | 3. Détection | Véhicules stationnés ou en circulation |
| `report.py` | 4. Rapport | `detections.csv` vers `report.csv` |
| `analysis.py` | 5. Analyse | Statistiques, bootstrap, graphiques |
| `evaluate.py` | Évaluation | Comparaison avec `annotations/annotations.csv` |

Fichiers produits pour chaque zone, dans `data/<zone>/` (non versionné) :
`images/`, `metadata.csv`, `filter.csv`, `detections.csv`, `report.csv`.
Les colonnes de chaque fichier sont définies dans `config.py` et dans la
section 5 du plan de travail.

## Règles du groupe

- Une branche par fonctionnalité, fusion dans `main` par pull request relue par un autre membre.
- Le token Mapillary reste dans `.env`, jamais dans le code.
- Code commenté en français, une docstring par fonction, annotations de type.
- Sources ajoutées dans `references.md` et usages d'IA générative dans `usage_ia.md`, au fur et à mesure.
- Tests : `python -m pytest`.

## État d'avancement

| Étape | État |
|---|---|
| 1. Collecte | Fonctionnelle (API Graph, liste figée par zone) |
| 2. Filtrage | Seulement `--filter none` ; BLIP-2 et CLIP à coder (R2) |
| 3. Détection | Fonctionnelle (YOLO, regroupement des classes, conducteurs de deux-roues) |
| 3. Stationnés | Seulement `--parked none` ; stratégies `geometry` et `clip` à coder (R4) |
| 4. Rapport | Fonctionnel |
| 5. Analyse | À coder (R5) |
| Évaluation | À coder (R4) |

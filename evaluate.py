"""Évaluation du pipeline sur le jeu annoté à la main (responsable : R4).

Compare les sorties d'une zone (filter.csv, detections.csv, report.csv) avec
annotations/annotations.csv (colonnes : config.ANNOTATION_COLUMNS).

Rappel (section 6.7 du plan) : 75 images servent à régler les seuils, 75 autres
à l'évaluation finale. Les images d'évaluation ne servent jamais à régler.

Exemple :
    python evaluate.py --zone data/carrefour_pistes --split test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config

log = logging.getLogger(__name__)

ANNOTATIONS_PATH = Path("annotations") / "annotations.csv"


# ---------------------------------------------------------------------------
# À implémenter (R4)
# ---------------------------------------------------------------------------
def evaluate_filter(filter_csv: Path, annotations_csv: Path, split: str) -> dict:
    """Précision, rappel et F1 de la décision "conserver"."""
    raise NotImplementedError("TODO (R4) : évaluation du filtre")


def evaluate_counts(report_csv: Path, annotations_csv: Path, split: str):
    """Erreur absolue moyenne et biais moyen par catégorie. Retourne un DataFrame."""
    raise NotImplementedError("TODO (R4) : évaluation des comptes")


def evaluate_parked(detections_csv: Path, annotations_csv: Path, split: str) -> dict:
    """Comparaison du nombre de véhicules stationnés prédits et annotés."""
    raise NotImplementedError("TODO (R4) : évaluation du tri des stationnés")


def main(argv: list[str] | None = None) -> int:
    """Affiche et enregistre les mesures de qualité d'une zone."""
    parser = argparse.ArgumentParser(description="Évaluation sur le jeu annoté.")
    parser.add_argument("--zone", type=Path, required=True, help="dossier de la zone")
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    parser.add_argument("--split", choices=("reglage", "test"), default="test")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")
    raise NotImplementedError("TODO (R4) : enchaîner les évaluations")


if __name__ == "__main__":
    sys.exit(main())

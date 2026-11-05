"""Étape 4 : génération de report.csv (responsable : R5).

Une ligne par image conservée par le filtre, avec le nombre d'objets en
circulation de chaque catégorie (colonnes : config.REPORT_COLUMNS).

Règles (section 6.5 du plan) :
    - un objet est compté s'il n'est pas un conducteur (is_rider faux)
      et n'est pas stationné (parked faux), sauf si keep_parked est vrai ;
    - une image conservée sans détection apparaît avec des zéros.
"""
from __future__ import annotations

import logging
from pathlib import Path

import config

log = logging.getLogger(__name__)


def build_report(zone_dir: Path, output: Path, keep_parked: bool = False) -> Path:
    """Lit filter.csv et detections.csv dans zone_dir, écrit report.csv dans output.

    Retourne le chemin du fichier écrit.
    """
    raise NotImplementedError("TODO (R5) : agrégation par image")

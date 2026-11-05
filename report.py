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

import pandas as pd

import config

log = logging.getLogger(__name__)


def _is_true(series: pd.Series) -> pd.Series:
    """Colonne booléenne lue depuis un CSV ("True", "False" ou vide)."""
    return series.fillna("").astype(str).str.strip().str.lower() == "true"


def _read(path: Path, step: str) -> pd.DataFrame:
    """Lit un CSV du pipeline en texte, avec un message clair s'il manque."""
    if not path.exists():
        raise RuntimeError(f"{path} introuvable : lancez d'abord l'étape {step}.")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def count_objects(detections: pd.DataFrame, image_ids: list[str],
                  keep_parked: bool = False) -> pd.DataFrame:
    """Tableau image_id x catégorie du nombre d'objets comptés.

    Toutes les images de image_ids et toutes les catégories de config.CATEGORIES
    sont présentes, avec 0 quand rien n'a été compté.
    """
    counted = detections[detections["image_id"].isin(image_ids)]
    counted = counted[~_is_true(counted["is_rider"])]
    if not keep_parked:
        counted = counted[~_is_true(counted["parked"])]
    table = (counted.groupby(["image_id", "category"]).size()
             .unstack(fill_value=0)
             .reindex(index=image_ids, columns=config.CATEGORIES, fill_value=0))
    return table.fillna(0).astype(int)


def build_report(zone_dir: Path, output: Path, keep_parked: bool = False) -> Path:
    """Lit filter.csv, metadata.csv et detections.csv dans zone_dir, écrit report.csv.

    Retourne le chemin du fichier écrit.
    """
    zone_dir = Path(zone_dir)
    filtered = _read(zone_dir / config.FILTER_FILE, "filter")
    metadata = _read(zone_dir / config.METADATA_FILE, "download")
    detections = _read(zone_dir / config.DETECTIONS_FILE, "detect")

    image_ids = filtered.loc[_is_true(filtered["keep"]), "image_id"].tolist()
    table = count_objects(detections, image_ids, keep_parked)

    # nom du fichier image (colonne demandée par le sujet)
    paths = metadata.set_index("image_id")["path"].to_dict()
    names = [Path(paths.get(i) or f"{i}.jpg").name for i in table.index]
    report = table.reset_index(drop=True)
    report.insert(0, "image", names)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report[config.REPORT_COLUMNS].to_csv(output, index=False, encoding="utf-8")

    totals = ", ".join(f"{c} {int(table[c].sum())}" for c in config.CATEGORIES)
    log.info("%d image(s) dans le rapport ; objets comptés : %s%s", len(report), totals,
             " (stationnés inclus)" if keep_parked else "")
    return output

"""Étape 5 : analyse statistique et graphiques (responsable : R5).

Se lance séparément, sur une ou plusieurs zones déjà traitées par main.py :
    python analysis.py --zones data/zone_a data/zone_b data/zone_c --out figures/

Produit stats.csv (colonnes : config.STATS_COLUMNS) et les graphiques de la
section 6.6 du plan.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# À implémenter (R5)
# ---------------------------------------------------------------------------
def compute_indicators(report_csv: Path, metadata_csv: Path, zone: str):
    """Calcule, pour chaque catégorie : nombre, part modale, taux de présence,
    moyenne par image. Retourne un DataFrame au format stats.csv (sans intervalles).
    """
    raise NotImplementedError("TODO (R5) : indicateurs d'une zone")


def bootstrap_shares(report_csv: Path, metadata_csv: Path,
                     n_boot: int = config.N_BOOTSTRAP,
                     level: float = config.CONFIDENCE_LEVEL,
                     seed: int = config.RANDOM_SEED):
    """Intervalles de confiance des parts modales par bootstrap.

    Les tirages se font sur les séquences (sequence_id) et non sur les images,
    car les images d'une même séquence sont corrélées.
    Retourne un DataFrame (category, share_low, share_high).
    """
    raise NotImplementedError("TODO (R5) : bootstrap par séquence")


def compare_zones(stats):
    """Test du khi-deux sur le tableau zone x catégorie (à interpréter avec prudence)."""
    raise NotImplementedError("TODO (R5) : comparaison des zones")


def plot_all(stats, zone_dirs: list[Path], out_dir: Path) -> None:
    """Trace les graphiques de la section 6.6 (parts modales, présence, carte, dates, ablation)."""
    raise NotImplementedError("TODO (R5) : graphiques")


def main(argv: list[str] | None = None) -> int:
    """Analyse comparée des zones passées en argument."""
    parser = argparse.ArgumentParser(description="Analyse comparée des zones traitées.")
    parser.add_argument("--zones", nargs="+", type=Path, required=True,
                        help="dossiers des zones (par exemple data/carrefour_pistes)")
    parser.add_argument("--out", type=Path, default=Path(config.FIGURES_DIR),
                        help="dossier de sortie des graphiques et de stats.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("TODO (R5) : enchaîner indicateurs, bootstrap, tests et graphiques")


if __name__ == "__main__":
    sys.exit(main())

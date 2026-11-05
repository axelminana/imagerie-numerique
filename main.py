"""Point d'entrée de l'application en ligne de commande.

Enchaîne les étapes du pipeline pour une zone :
    1. download : collecte des images Mapillary      -> metadata.csv, images/
    2. filter   : filtrage vision-langage            -> filter.csv
    3. detect   : détection YOLO et tri des stationnés -> detections.csv
    4. report   : agrégation par image               -> report.csv

L'analyse comparée des zones (étape 5) se lance séparément avec analysis.py.

Exemple :
    python main.py -lat 45.800413 -lon 3.108813 -o report.csv --zone carrefour_pistes
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config

ALL_STEPS = ("download", "filter", "detect", "report")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lit les options de la ligne de commande (section 4 du plan)."""
    parser = argparse.ArgumentParser(
        description="Estimation des modalités de mobilité à partir d'images Mapillary.",
    )
    # Options imposées par le sujet (argparse fournit -h automatiquement)
    parser.add_argument("-lat", type=float, required=True,
                        help="latitude du centre de la zone")
    parser.add_argument("-lon", type=float, required=True,
                        help="longitude du centre de la zone")
    parser.add_argument("-o", dest="output", default=config.REPORT_FILE,
                        help="nom du fichier CSV de résultats (défaut : report.csv)")

    # Options supplémentaires
    parser.add_argument("--radius", type=float, default=config.DEFAULT_RADIUS_M,
                        help="rayon en mètres (défaut : %(default)s)")
    parser.add_argument("--zone", default=None,
                        help="nom du dossier de travail dans data/ (défaut : déduit de lat/lon)")
    parser.add_argument("--spacing", type=float, default=config.DEFAULT_SPACING_M,
                        help="distance minimale en mètres entre deux images d'une séquence")
    parser.add_argument("--max-images", type=int, default=None,
                        help="nombre maximal d'images (tests rapides)")
    parser.add_argument("--filter", dest="filter_method", default="clip",
                        choices=("blip2", "clip", "none"),
                        help="méthode de filtrage des images (défaut provisoire : clip)")
    parser.add_argument("--detector", default=config.DEFAULT_DETECTOR,
                        help="poids YOLO à utiliser (défaut : %(default)s)")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONF,
                        help="seuil de confiance YOLO (défaut : %(default)s)")
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ,
                        help="taille d'entrée YOLO en pixels (défaut : %(default)s)")
    parser.add_argument("--parked", dest="parked_strategy",
                        default=config.DEFAULT_PARKED_STRATEGY,
                        choices=config.PARKED_STRATEGIES,
                        help="stratégie de détection des véhicules stationnés")
    parser.add_argument("--keep-parked", action="store_true",
                        help="compter aussi les véhicules stationnés dans le rapport")
    parser.add_argument("--skip-download", action="store_true",
                        help="réutiliser les images déjà téléchargées")
    parser.add_argument("--steps", default=",".join(ALL_STEPS),
                        help="étapes à exécuter, séparées par des virgules "
                             "(défaut : %(default)s)")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                        help="matériel de calcul (défaut : cuda)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="journal détaillé")

    args = parser.parse_args(argv)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        parser.error(f"étape(s) inconnue(s) : {', '.join(unknown)}. "
                     f"Choix possibles : {', '.join(ALL_STEPS)}")
    args.steps = steps
    if args.skip_download and "download" in args.steps:
        args.steps.remove("download")
    return args


def zone_name(lat: float, lon: float) -> str:
    """Nom de dossier par défaut d'une zone, construit à partir des coordonnées."""
    return f"zone_{lat:.5f}_{lon:.5f}".replace("-", "m")


def main(argv: list[str] | None = None) -> int:
    """Exécute les étapes demandées et retourne le code de sortie du programme."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s : %(message)s",
    )
    log = logging.getLogger("main")

    zone_dir = config.DATA_DIR / (args.zone or zone_name(args.lat, args.lon))
    zone_dir.mkdir(parents=True, exist_ok=True)
    log.info("Zone : %s (lat=%s, lon=%s, rayon=%s m)", zone_dir, args.lat, args.lon, args.radius)

    # Les modules sont importés au moment de l'étape : "python main.py -h"
    # fonctionne ainsi sans que torch ou ultralytics soient installés.
    if "download" in args.steps:
        import mapillary_client
        mapillary_client.collect(
            lat=args.lat, lon=args.lon, radius_m=args.radius, zone_dir=zone_dir,
            spacing_m=args.spacing, max_images=args.max_images,
        )

    if "filter" in args.steps:
        import filtering
        filtering.filter_images(zone_dir=zone_dir, method=args.filter_method, device=args.device)

    if "detect" in args.steps:
        import detection
        detection.detect(
            zone_dir=zone_dir, model_path=args.detector, conf=args.conf,
            imgsz=args.imgsz, device=args.device, parked_strategy=args.parked_strategy,
        )

    if "report" in args.steps:
        import report
        output = Path(args.output)
        if not output.is_absolute() and output.parent == Path("."):
            output = zone_dir / output
        report.build_report(zone_dir=zone_dir, output=output, keep_parked=args.keep_parked)
        log.info("Rapport écrit dans %s", output)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Préparation et regroupement des annotations à la main (section 6.7 du plan).

1. Préparer : répartit les images téléchargées des zones entre les annotateurs.
    python prepare_annotation.py --zones data/carrefour_pistes data/damsplatz --annotateurs A B C

   Crée le dossier annotation/ avec :
    - un sous-dossier d'images par annotateur, numérotées dans l'ordre de sa feuille ;
    - une feuille par annotateur (feuille_<nom>.csv, à ouvrir dans Excel) ;
    - consigne.txt, les règles communes d'annotation.
   Chaque zone est coupée en deux moitiés tirées au hasard : "reglage" (pour
   choisir les seuils) et "test" (réservée à l'évaluation finale). Les images
   de chaque zone sont réparties à parts égales entre les annotateurs, et
   quelques images (--doubles) sont confiées en plus à un second annotateur
   pour mesurer l'accord entre annotateurs.

2. Regrouper : une fois les feuilles remplies, les réunit dans annotations/annotations.csv.
    python prepare_annotation.py --merge
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import sys
from pathlib import Path

import config

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("annotation")
ANNOTATIONS_PATH = Path("annotations") / "annotations.csv"
# Colonnes des feuilles : d'abord ce qui sert à retrouver l'image, puis ce qu'on remplit.
SHEET_COLUMNS: list[str] = (
    ["fichier", "zone", "image_id", "split", "double", "annotateur", "keep"]
    + config.CATEGORIES + ["stationnes", "commentaire"]
)
EXCEL_DELIMITER = ";"            # séparateur attendu par Excel en français
EXCEL_ENCODING = "utf-8-sig"     # UTF-8 avec signature : accents lus correctement par Excel

CONSIGNE = """CONSIGNE D'ANNOTATION
=====================

Remplir sa feuille (feuille_<nom>.csv) en ouvrant chaque image de son dossier.
Annoter les images brutes : ne jamais regarder les images avec les boîtes de YOLO.
Compter ce qu'on identifie à taille d'affichage normale, sans zoomer.

keep : oui ou non
  oui : photo prise depuis l'espace public, montrant une voie ouverte à la
        circulation (rue, route, piste cyclable, trottoir le long d'une
        chaussée), MÊME VIDE. Une vue à travers le pare-brise est gardée.
  non : parking, intérieur du véhicule qui occupe l'essentiel de l'image,
        photo trop floue ou trop sombre pour compter, façade ou champ sans chaussée.
  Si non : laisser les comptes vides.

Comptes des usagers EN CIRCULATION (nombres entiers, 0 si aucun) :
  pietons               toute personne à pied, même immobile
  deux_roues            vélos ; un vélo avec son cycliste compte pour 1 vélo,
                        le cycliste n'est PAS compté comme piéton
  deux_roues_motorises  motos et scooters, même règle pour le conducteur
  vehicules_legers      voitures, taxis
  bus                   bus et cars
  camions               camions, mais aussi camionnettes et utilitaires
                        (la classe "truck" de YOLO les regroupe)
  Trams : ne pas les compter, écrire "tram" dans commentaire.
  Ne pas compter : le véhicule d'où la photo est prise (capot visible),
  les reflets, les affiches, les personnes à l'intérieur des véhicules.

En circulation ou stationné :
  en circulation : sur la voie, y compris arrêté à un feu ou dans un bouchon
  stationné      : garé sur une place, le long du trottoir, dans une file de
                   véhicules garés ; vélos et motos attachés ou sur béquille
  stationnes : nombre TOTAL de véhicules garés, toutes catégories confondues.
  Un véhicule garé ne va PAS dans sa catégorie.

En cas de doute : décider, et l'écrire dans commentaire.
Les images marquées double = oui sont aussi annotées par un autre membre :
ne pas se concerter avant d'avoir fini.
"""


def selected_images(zone_dir: Path) -> list[dict]:
    """Images téléchargées et retenues d'une zone (selected vrai dans metadata.csv)."""
    metadata_path = zone_dir / config.METADATA_FILE
    if not metadata_path.exists():
        raise RuntimeError(f"{metadata_path} introuvable : lancez d'abord l'étape download.")
    images = []
    with open(metadata_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["selected"].strip().lower() != "true" or not row["path"]:
                continue
            path = zone_dir / row["path"]
            if path.exists():
                images.append({"image_id": row["image_id"], "zone": zone_dir.name, "source": path})
            else:
                log.warning("Image absente du disque : %s", path)
    return sorted(images, key=lambda img: img["image_id"])


def assign(images_by_zone: dict[str, list[dict]], annotators: list[str],
           n_doubles: int, seed: int = config.RANDOM_SEED) -> list[dict]:
    """Répartition des images : une ligne par image et par annotateur.

    Dans chaque zone : moitié "reglage", moitié "test", puis répartition à tour
    de rôle entre les annotateurs (chacun reçoit des images des deux moitiés
    et de toutes les zones). Ensuite n_doubles images, prises à tour de rôle
    dans les zones, reçoivent un second annotateur, différent du premier.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for zone in sorted(images_by_zone):
        images = list(images_by_zone[zone])
        rng.shuffle(images)
        half = len(images) // 2
        for index, image in enumerate(images):
            image["split"] = "reglage" if index < half else "test"
            image["annotateur"] = annotators[index % len(annotators)]
        rows.extend(images)

    # doubles : alternance entre zones pour qu'elles soient toutes représentées
    pools = [list(images_by_zone[zone]) for zone in sorted(images_by_zone)]
    for pool in pools:
        rng.shuffle(pool)
    doubles: list[dict] = []
    while len(doubles) < n_doubles and any(pools):
        for pool in pools:
            if pool and len(doubles) < n_doubles:
                doubles.append(pool.pop())
    if len(doubles) < n_doubles:
        log.warning("Seulement %d image(s) disponibles pour la double annotation", len(doubles))

    sheet_rows = [{**image, "double": "non"} for image in rows]
    double_ids = {image["image_id"] for image in doubles}
    for row in sheet_rows:
        if row["image_id"] in double_ids:
            row["double"] = "oui"
    load = {name: sum(row["annotateur"] == name for row in sheet_rows) for name in annotators}
    for image in doubles:
        # second annotateur : le membre le moins chargé parmi les autres
        others = [name for name in annotators if name != image["annotateur"]]
        second = min(others, key=lambda name: (load[name], annotators.index(name)))
        load[second] += 1
        sheet_rows.append({**image, "annotateur": second, "double": "oui"})
    return sheet_rows


def write_sheets(rows: list[dict], annotators: list[str], out_dir: Path) -> None:
    """Copie les images dans un dossier par annotateur et écrit sa feuille."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in annotators:
        mine = [row for row in rows if row["annotateur"] == name]
        random.Random(f"{config.RANDOM_SEED}-{name}").shuffle(mine)   # ordre mélangé
        folder = out_dir / name
        folder.mkdir(exist_ok=True)
        sheet = []
        for number, row in enumerate(mine, start=1):
            file_name = f"{number:03d}_{row['image_id']}.jpg"
            shutil.copy2(row["source"], folder / file_name)
            sheet.append({"fichier": f"{name}/{file_name}", "zone": row["zone"],
                          "image_id": row["image_id"], "split": row["split"],
                          "double": row["double"], "annotateur": name})
        with open(out_dir / f"feuille_{name}.csv", "w", newline="", encoding=EXCEL_ENCODING) as handle:
            writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS, delimiter=EXCEL_DELIMITER)
            writer.writeheader()
            writer.writerows(sheet)
        log.info("%s : %d image(s) dont %d en double", name, len(sheet),
                 sum(r["double"] == "oui" for r in sheet))
    (out_dir / "consigne.txt").write_text(CONSIGNE, encoding="utf-8")


def prepare(zone_dirs: list[Path], annotators: list[str], out_dir: Path = DEFAULT_OUT,
            n_doubles: int = 20, seed: int = config.RANDOM_SEED) -> list[dict]:
    """Prépare les dossiers et les feuilles d'annotation ; retourne la répartition."""
    if len(set(annotators)) != len(annotators) or len(annotators) < 2:
        raise RuntimeError("Il faut au moins deux annotateurs, avec des noms différents.")
    bad = [a for a in annotators if not a.replace("-", "").replace("_", "").isalnum()]
    if bad:
        raise RuntimeError(f"Noms d'annotateurs à éviter (lettres, chiffres, - et _ seulement) : {bad}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"{out_dir} existe déjà et n'est pas vide : supprimez-le ou choisissez --out.")
    images_by_zone = {}
    for zone_dir in zone_dirs:
        images = selected_images(Path(zone_dir))
        if not images:
            raise RuntimeError(f"Aucune image téléchargée dans {zone_dir}.")
        images_by_zone[Path(zone_dir).name] = images
        log.info("%s : %d image(s)", Path(zone_dir).name, len(images))
    rows = assign(images_by_zone, annotators, n_doubles, seed)
    write_sheets(rows, annotators, out_dir)
    log.info("Dossiers et feuilles dans %s ; lire %s", out_dir, out_dir / "consigne.txt")
    return rows


def merge(out_dir: Path = DEFAULT_OUT, output: Path = ANNOTATIONS_PATH) -> Path:
    """Réunit les feuilles remplies dans annotations.csv (colonnes : config.ANNOTATION_COLUMNS).

    Accepte les feuilles enregistrées par Excel avec ; ou par Google Sheets avec ,.
    Signale les lignes incomplètes sans les rejeter.
    """
    sheets = sorted(out_dir.glob("feuille_*.csv"))
    if not sheets:
        raise RuntimeError(f"Aucune feuille feuille_*.csv dans {out_dir}.")
    rows = []
    for sheet in sheets:
        text = sheet.read_text(encoding="utf-8-sig", errors="replace")
        delimiter = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
        for row in csv.DictReader(text.splitlines(), delimiter=delimiter):
            keep = (row.get("keep") or "").strip().lower()
            if keep not in ("oui", "non"):
                log.warning("%s, %s : keep vaut %r (attendu oui ou non)", sheet.name, row.get("fichier"), keep)
            elif keep == "oui" and any(not (row.get(c) or "").strip() for c in config.CATEGORIES):
                log.warning("%s, %s : comptes incomplets", sheet.name, row.get("fichier"))
            rows.append({c: (row.get(c) or "").strip() for c in config.ANNOTATION_COLUMNS})
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("%d ligne(s) de %d feuille(s) dans %s", len(rows), len(sheets), output)
    return output


def main(argv: list[str] | None = None) -> int:
    """Prépare les annotations (--zones) ou regroupe les feuilles remplies (--merge)."""
    parser = argparse.ArgumentParser(description="Préparation et regroupement des annotations.")
    parser.add_argument("--zones", nargs="+", type=Path, help="dossiers des zones à annoter")
    parser.add_argument("--annotateurs", nargs="+", default=["A", "B", "C"],
                        help="noms des annotateurs (défaut : A B C)")
    parser.add_argument("--doubles", type=int, default=20,
                        help="images annotées deux fois (défaut : %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="dossier de préparation (défaut : %(default)s)")
    parser.add_argument("--merge", action="store_true",
                        help="regrouper les feuilles remplies dans annotations/annotations.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")
    try:
        if args.merge:
            merge(args.out)
        elif args.zones:
            prepare(args.zones, args.annotateurs, args.out, args.doubles)
        else:
            parser.error("indiquer --zones pour préparer, ou --merge pour regrouper")
    except RuntimeError as error:
        log.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Étape 5 : analyse statistique et graphiques (responsable : R5).

Se lance séparément, sur une ou plusieurs zones déjà traitées par main.py :
    python analysis.py --zones data/carrefour_pistes data/damsplatz --out figures/

Produit dans le dossier de sortie (section 6.6 du plan) :
    - stats.csv : une ligne par zone et par catégorie (colonnes : config.STATS_COLUMNS),
      avec la part modale et son intervalle de confiance à 95 % ;
    - comparaison.txt : test du khi-deux entre les zones et V de Cramér ;
    - parts_modales.png : parts modales par zone, barres empilées à 100 % ;
    - comparaison_zones.png : parts modales avec intervalles de confiance et
      taux de présence, catégorie par catégorie ;
    - carte_<zone>.png : position des images de chaque zone ;
    - dates.png : années de prise de vue des images retenues.

Les intervalles de confiance sont obtenus par bootstrap sur les séquences et
non sur les images, car les images d'une même séquence (même trajet, même
moment) sont corrélées : un même véhicule peut y apparaître plusieurs fois.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

TEXT_DARK = "#191919"       # encre principale
TEXT_MUTED = "#666666"      # encre secondaire (axes, notes)
GRID = "#e5e5e5"
ZONE_STYLES = [             # identité des zones : couleur et forme du marqueur
    ("#6a4ce2", "o"),
    ("#ec4647", "s"),
    ("#191919", "^"),
    ("#666666", "D"),
]


# ---------------------------------------------------------------------------
# Lecture des données d'une zone
# ---------------------------------------------------------------------------
def _read(path: Path, step: str) -> pd.DataFrame:
    """Lit un CSV du pipeline en texte, avec un message clair s'il manque."""
    if not path.exists():
        raise RuntimeError(f"{path} introuvable : lancez d'abord l'étape {step}.")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_zone(report_csv: Path, metadata_csv: Path) -> pd.DataFrame:
    """Une ligne par image du rapport : image_id, sequence_id et un compte par catégorie.

    L'identifiant de l'image est le nom du fichier sans extension ; la séquence
    vient de metadata.csv (une image sans séquence connue forme sa propre séquence).
    """
    report = _read(Path(report_csv), "report")
    metadata = _read(Path(metadata_csv), "download")
    missing = [c for c in config.REPORT_COLUMNS if c not in report.columns]
    if missing:
        raise RuntimeError(f"{report_csv} : colonnes manquantes {missing}")

    data = pd.DataFrame({"image_id": report["image"].map(lambda name: Path(name).stem)})
    for category in config.CATEGORIES:
        data[category] = pd.to_numeric(report[category], errors="coerce").fillna(0).astype(int)
    sequences = metadata.set_index("image_id")["sequence_id"].to_dict()
    data["sequence_id"] = [sequences.get(i) or f"image_{i}" for i in data["image_id"]]
    return data


# ---------------------------------------------------------------------------
# Indicateurs
# ---------------------------------------------------------------------------
def indicators(data: pd.DataFrame, zone: str) -> pd.DataFrame:
    """Nombre, part modale, taux de présence et moyenne par image de chaque catégorie.

    La part modale est la part de la catégorie parmi tous les objets comptés ;
    le taux de présence est la part des images qui contiennent au moins un
    objet de la catégorie. Les colonnes share_low et share_high restent vides
    (remplies par bootstrap_shares).
    """
    counts = data[config.CATEGORIES]
    total = int(counts.to_numpy().sum())
    n_images = len(data)
    rows = []
    for category in config.CATEGORIES:
        count = int(counts[category].sum())
        rows.append({
            "zone": zone,
            "category": category,
            "count": count,
            "share": round(count / total, 4) if total else float("nan"),
            "share_low": float("nan"),
            "share_high": float("nan"),
            "presence_rate": round(float((counts[category] > 0).mean()), 4) if n_images else float("nan"),
            "mean_per_image": round(float(counts[category].mean()), 4) if n_images else float("nan"),
        })
    return pd.DataFrame(rows, columns=config.STATS_COLUMNS)


def compute_indicators(report_csv: Path, metadata_csv: Path, zone: str) -> pd.DataFrame:
    """Indicateurs d'une zone à partir de ses fichiers (voir indicators)."""
    return indicators(load_zone(report_csv, metadata_csv), zone)


def bootstrap_shares(data: pd.DataFrame, n_boot: int = config.N_BOOTSTRAP,
                     level: float = config.CONFIDENCE_LEVEL,
                     seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Intervalles de confiance des parts modales par bootstrap sur les séquences.

    À chaque tirage, on tire avec remise autant de séquences qu'il y en a, on
    additionne leurs comptes et on recalcule les parts. Les bornes sont les
    quantiles (1 - level) / 2 et (1 + level) / 2 des parts obtenues.
    Retourne un DataFrame (category, share_low, share_high).
    """
    by_sequence = data.groupby("sequence_id")[config.CATEGORIES].sum().to_numpy(dtype=float)
    n_seq = len(by_sequence)
    if n_seq < 2:
        log.warning("Moins de deux séquences : intervalle de confiance impossible")
        return pd.DataFrame({"category": config.CATEGORIES,
                             "share_low": float("nan"), "share_high": float("nan")})
    if n_seq < 10:
        log.warning("Seulement %d séquences : intervalles de confiance peu fiables", n_seq)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_seq, size=(n_boot, n_seq))
    sums = by_sequence[draws].sum(axis=1)                 # (n_boot, catégories)
    totals = sums.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.where(totals > 0, sums / totals, np.nan)
    alpha = (1 - level) / 2
    low = np.nanquantile(shares, alpha, axis=0)
    high = np.nanquantile(shares, 1 - alpha, axis=0)
    return pd.DataFrame({"category": config.CATEGORIES,
                         "share_low": np.round(low, 4), "share_high": np.round(high, 4)})


def zone_stats(data: pd.DataFrame, zone: str, **bootstrap_options) -> pd.DataFrame:
    """Indicateurs d'une zone complétés par les intervalles de confiance."""
    stats = indicators(data, zone).drop(columns=["share_low", "share_high"])
    stats = stats.merge(bootstrap_shares(data, **bootstrap_options), on="category")
    return stats[config.STATS_COLUMNS]


def compare_zones(stats: pd.DataFrame) -> dict:
    """Test du khi-deux d'indépendance sur le tableau zone x catégorie.

    Les catégories absentes de toutes les zones sont retirées. Le test suppose
    des observations indépendantes, ce qui n'est pas le cas ici (objets d'une
    même image ou d'une même séquence) : la p-valeur est donc trop optimiste et
    s'interprète avec prudence ; les intervalles de confiance restent
    l'argument principal. Le V de Cramér mesure l'intensité de l'écart (0 : mêmes
    répartitions, 1 : répartitions totalement différentes).
    """
    from scipy.stats import chi2_contingency

    table = stats.pivot(index="zone", columns="category", values="count").fillna(0)
    table = table.loc[:, table.sum(axis=0) > 0]
    table = table.loc[table.sum(axis=1) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"ok": False, "message": "au moins deux zones et deux catégories non vides sont nécessaires"}

    chi2, p_value, dof, expected = chi2_contingency(table.to_numpy())
    n = table.to_numpy().sum()
    cramer_v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"ok": True, "chi2": float(chi2), "p_value": float(p_value), "dof": int(dof),
            "cramer_v": float(cramer_v), "n_objects": int(n),
            "small_expected": int((expected < 5).sum()), "table": table.astype(int)}


def format_comparison(result: dict) -> str:
    """Texte lisible du résultat de compare_zones (écrit dans comparaison.txt)."""
    if not result["ok"]:
        return f"Comparaison impossible : {result['message']}\n"
    lines = [
        "Comparaison des zones : test du khi-deux d'indépendance",
        "",
        result["table"].rename(columns=config.CATEGORY_LABELS).to_string(),
        "",
        f"khi-deux = {result['chi2']:.1f}, ddl = {result['dof']}, p = {result['p_value']:.2g}",
        f"V de Cramér = {result['cramer_v']:.2f} ({result['n_objects']} objets)",
        "",
        "Prudence : les objets d'une même image ou d'une même séquence ne sont pas",
        "indépendants, la p-valeur est donc trop optimiste. S'appuyer d'abord sur",
        "les intervalles de confiance de stats.csv.",
    ]
    if result["small_expected"]:
        lines.append(f"{result['small_expected']} case(s) ont un effectif attendu inférieur à 5 : "
                     "le test est peu fiable pour ces catégories.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------
def _style_axes(ax) -> None:
    """Axes discrets : pas de cadre haut et droit, grille légère, encre secondaire."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_DARK, length=0)
    ax.set_axisbelow(True)


def _label_color(hex_color: str) -> str:
    """Encre lisible (blanche ou foncée) sur un segment de couleur hex_color."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return "#ffffff" if luminance < 0.3 else TEXT_DARK


def plot_shares(stats: pd.DataFrame, path: Path, note: str = "") -> None:
    """Parts modales par zone : une barre horizontale empilée à 100 % par zone.

    Les segments d'au moins 5 % portent leur pourcentage ; les autres sont lisibles
    dans la légende et dans stats.csv.
    """
    import matplotlib.pyplot as plt

    zones = list(dict.fromkeys(stats["zone"]))
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.75 * len(zones)), dpi=150)
    for row, zone in enumerate(zones):
        zone_rows = stats[stats["zone"] == zone].set_index("category")
        left = 0.0
        for category in config.CATEGORIES:
            share = float(zone_rows.loc[category, "share"]) if category in zone_rows.index else 0.0
            if not share or math.isnan(share):
                continue
            color = config.CATEGORY_COLORS[category]
            ax.barh(row, share, left=left, height=0.6, color=color,
                    edgecolor="#ffffff", linewidth=2,
                    label=config.CATEGORY_LABELS[category] if row == 0 else None)
            if share >= 0.05:
                ax.text(left + share / 2, row, f"{share:.0%}", ha="center", va="center",
                        fontsize=9, color=_label_color(color))
            left += share
    # légende complète, dans l'ordre fixe des catégories
    handles = [plt.Rectangle((0, 0), 1, 1, color=config.CATEGORY_COLORS[c]) for c in config.CATEGORIES]
    ax.legend(handles, [config.CATEGORY_LABELS[c] for c in config.CATEGORIES],
              ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25 if len(zones) > 1 else -0.45),
              fontsize=9, labelcolor=TEXT_DARK)
    ax.set_yticks(range(len(zones)), zones)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _style_axes(ax)
    ax.set_title("Part modale estimée par zone", loc="left", fontsize=12, color=TEXT_DARK)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(stats: pd.DataFrame, path: Path) -> None:
    """Deux panneaux, catégorie par catégorie : part modale avec intervalle de
    confiance à 95 %, et taux de présence. Une zone = une couleur et une forme."""
    import matplotlib.pyplot as plt

    zones = list(dict.fromkeys(stats["zone"]))
    labels = [config.CATEGORY_LABELS[c] for c in config.CATEGORIES]
    y = np.arange(len(config.CATEGORIES))
    offsets = np.linspace(-0.25, 0.25, len(zones)) if len(zones) > 1 else [0.0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, sharey=True)
    for (ax, title) in zip(axes, ("Part modale (intervalle de confiance à 95 %)",
                                  "Images où la catégorie est présente")):
        ax.set_title(title, loc="left", fontsize=10.5, color=TEXT_DARK)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        _style_axes(ax)

    for index, zone in enumerate(zones):
        color, marker = ZONE_STYLES[index % len(ZONE_STYLES)]
        rows = stats[stats["zone"] == zone].set_index("category").reindex(config.CATEGORIES)
        share = rows["share"].astype(float).to_numpy()
        low = rows["share_low"].astype(float).to_numpy()
        high = rows["share_high"].astype(float).to_numpy()
        ypos = y + offsets[index]
        has_ci = ~np.isnan(low) & ~np.isnan(high)
        axes[0].hlines(ypos[has_ci], low[has_ci], high[has_ci], color=color, linewidth=2)
        axes[0].scatter(share, ypos, color=color, marker=marker, s=40, zorder=3,
                        edgecolor="#ffffff", linewidth=1.5, label=zone)
        axes[1].scatter(rows["presence_rate"].astype(float), ypos, color=color, marker=marker,
                        s=40, zorder=3, edgecolor="#ffffff", linewidth=1.5)

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlim(left=0)
    axes[1].set_xlim(0, 1)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT_DARK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_map(zone_dir: Path, zone: str, path: Path) -> None:
    """Position des images d'une zone : candidates, gardées par le filtre, rejetées."""
    import matplotlib.pyplot as plt

    metadata = _read(zone_dir / config.METADATA_FILE, "download")
    lat = pd.to_numeric(metadata["lat"], errors="coerce")
    lon = pd.to_numeric(metadata["lon"], errors="coerce")
    selected = metadata["selected"].str.lower() == "true"
    keep = pd.Series(False, index=metadata.index)
    filter_path = zone_dir / config.FILTER_FILE
    if filter_path.exists():
        filtered = _read(filter_path, "filter")
        kept_ids = set(filtered.loc[filtered["keep"].str.lower() == "true", "image_id"])
        keep = metadata["image_id"].isin(kept_ids)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    ax.scatter(lon[~selected], lat[~selected], s=6, color="#c8c8c8", label="non retenues", zorder=1)
    ax.scatter(lon[selected & keep], lat[selected & keep], s=18, color="#6a4ce2",
               edgecolor="#ffffff", linewidth=0.8, label="gardées par le filtre", zorder=3)
    ax.scatter(lon[selected & ~keep], lat[selected & ~keep], s=22, color=TEXT_DARK, marker="x",
               linewidth=1.2, label="rejetées par le filtre", zorder=4)
    mean_lat = float(lat.mean()) if lat.notna().any() else 0.0
    ax.set_aspect(1 / math.cos(math.radians(mean_lat)))   # mètres égaux dans les deux sens
    ax.set_xlabel("longitude", color=TEXT_MUTED, fontsize=9)
    ax.set_ylabel("latitude", color=TEXT_MUTED, fontsize=9)
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(labelsize=8)
    _style_axes(ax)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_title(f"Images Mapillary : {zone}", loc="left", fontsize=11, color=TEXT_DARK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              labelcolor=TEXT_DARK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dates(zone_dirs: list[Path], zones: list[str], path: Path) -> None:
    """Nombre d'images retenues par année de prise de vue, une ligne de graphique par zone."""
    import matplotlib.pyplot as plt

    years_by_zone = {}
    for zone_dir, zone in zip(zone_dirs, zones):
        metadata = _read(zone_dir / config.METADATA_FILE, "download")
        chosen = metadata[metadata["selected"].str.lower() == "true"]
        years = pd.to_datetime(chosen["captured_at"], errors="coerce", utc=True).dt.year.dropna()
        years_by_zone[zone] = years.astype(int).value_counts().sort_index()
    all_years = sorted({y for counts in years_by_zone.values() for y in counts.index})
    if not all_years:
        log.warning("Aucune date de prise de vue : graphique des dates ignoré")
        return

    fig, axes = plt.subplots(len(zones), 1, figsize=(8, 1.4 + 1.3 * len(zones)), dpi=150,
                             sharex=True, squeeze=False)
    for ax, zone in zip(axes[:, 0], zones):
        counts = years_by_zone[zone].reindex(all_years, fill_value=0)
        ax.bar(counts.index, counts.to_numpy(), width=0.7, color="#6a4ce2")
        ax.set_title(zone, loc="left", fontsize=10, color=TEXT_DARK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        _style_axes(ax)
    axes[-1, 0].set_xticks(all_years, [str(y) for y in all_years])
    fig.suptitle("Images retenues par année de prise de vue", x=0.01, ha="left",
                 fontsize=12, color=TEXT_DARK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_all(stats: pd.DataFrame, zone_dirs: list[Path], out_dir: Path,
             zones: list[str] | None = None) -> list[Path]:
    """Trace tous les graphiques de la section 6.6 et retourne leurs chemins."""
    import matplotlib
    matplotlib.use("Agg")                    # pas de fenêtre : fichiers seulement

    zones = zones or list(dict.fromkeys(stats["zone"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "parts_modales.png", out_dir / "comparaison_zones.png"]
    n_images = {z: int(stats.loc[stats["zone"] == z, "n_images"].iloc[0])
                for z in zones} if "n_images" in stats else {}
    note = ", ".join(f"{z} : {n} images" for z, n in n_images.items())
    plot_shares(stats, paths[0], note=note)
    plot_comparison(stats, paths[1])
    for zone_dir, zone in zip(zone_dirs, zones):
        path = out_dir / f"carte_{zone}.png"
        plot_map(zone_dir, zone, path)
        paths.append(path)
    dates_path = out_dir / "dates.png"
    plot_dates(zone_dirs, zones, dates_path)
    if dates_path.exists():
        paths.append(dates_path)
    return paths


# ---------------------------------------------------------------------------
# Enchaînement
# ---------------------------------------------------------------------------
def analyse(zone_dirs: list[Path], out_dir: Path, names: list[str] | None = None,
            report_name: str = config.REPORT_FILE) -> pd.DataFrame:
    """Analyse complète : stats.csv, comparaison.txt et graphiques dans out_dir.

    names donne le nom affiché de chaque zone (par défaut, le nom de son dossier).
    Retourne le tableau des statistiques.
    """
    zone_dirs = [Path(z) for z in zone_dirs]
    names = names or [z.name for z in zone_dirs]
    if len(names) != len(zone_dirs):
        raise RuntimeError("--names doit donner autant de noms que --zones")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    image_counts = {}
    for zone_dir, name in zip(zone_dirs, names):
        data = load_zone(zone_dir / report_name, zone_dir / config.METADATA_FILE)
        image_counts[name] = len(data)
        tables.append(zone_stats(data, name))
        log.info("%s : %d image(s), %d séquence(s), %d objet(s)", name, len(data),
                 data["sequence_id"].nunique(), int(data[config.CATEGORIES].to_numpy().sum()))
    stats = pd.concat(tables, ignore_index=True)
    stats.to_csv(out_dir / config.STATS_FILE, index=False, encoding="utf-8")

    if len(zone_dirs) > 1:
        text = format_comparison(compare_zones(stats))
        (out_dir / "comparaison.txt").write_text(text, encoding="utf-8")
        log.info("\n%s", text)

    plot_stats = stats.assign(n_images=stats["zone"].map(image_counts))
    for path in plot_all(plot_stats, zone_dirs, out_dir, names):
        log.info("Graphique : %s", path)
    return stats


def main(argv: list[str] | None = None) -> int:
    """Analyse comparée des zones passées en argument."""
    parser = argparse.ArgumentParser(description="Analyse comparée des zones traitées.")
    parser.add_argument("--zones", nargs="+", type=Path, required=True,
                        help="dossiers des zones (par exemple data/carrefour_pistes)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="noms affichés des zones, dans le même ordre que --zones")
    parser.add_argument("--out", type=Path, default=Path(config.FIGURES_DIR),
                        help="dossier de sortie des graphiques et de stats.csv")
    parser.add_argument("--report", default=config.REPORT_FILE,
                        help="nom du rapport dans chaque zone (défaut : %(default)s)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")
    try:
        analyse(args.zones, args.out, args.names, args.report)
    except RuntimeError as error:
        log.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Étape 2 : filtrage des images avec un modèle vision-langage (responsable : R2).

Garde uniquement les images qui montrent une scène de circulation exploitable
(définition en section 6.2 du plan). Deux méthodes sont comparées :
    - BLIP-2 (demandé par le sujet) : on pose une question fermée sur l'image et
      on garde l'image si la réponse commence par "yes" ; en variante, une
      seconde question ("Is this a parking lot?") peut écarter l'image ;
    - CLIP (alternative plus légère) : on compare l'image à des descriptions
      positives et négatives ; le score est la probabilité totale des
      descriptions positives, et l'image est gardée au-dessus d'un seuil.

Produit filter.csv (colonnes : config.FILTER_COLUMNS).

Pour régler le filtre, le mode « variantes » calcule en une fois les réponses
aux trois questions BLIP-2, à la question de rejet et le score CLIP de chaque
image, dans filter_variants.csv, à comparer ensuite au jeu annoté :
    python filtering.py --zone data/carrefour_pistes --variants
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import config
from utils import batched, dtype_kwargs, free_gpu_memory, load_rgb, resolve_device

log = logging.getLogger(__name__)

FILTER_METHODS = ("blip2", "clip", "none")
VARIANT_COLUMNS: list[str] = (
    ["image_id"]
    + [f"blip2_q{i}" for i in range(len(config.BLIP2_PROMPTS))]
    + ["blip2_reject", "clip_score"]
)


# ---------------------------------------------------------------------------
# Règles de décision
# ---------------------------------------------------------------------------
def parse_yes_no(answer: str) -> bool:
    """Règle de décision de BLIP-2 : True si la réponse nettoyée commence par "yes"."""
    return answer.strip().lower().startswith("yes")


def clean_answer(decoded: str) -> str:
    """Ne garde que la réponse du modèle.

    Selon la version de transformers, le texte généré par BLIP-2 contient ou non
    la question : on garde ce qui suit le dernier "Answer:" puis la première ligne.
    """
    text = decoded.split("Answer:")[-1].strip()
    return text.splitlines()[0].strip() if text else ""


def blip2_keep(answer: str, reject_answer: str | None = None) -> bool:
    """Décision BLIP-2 : réponse "yes" à la question principale et, si la question
    de rejet a été posée, réponse autre que "yes" à celle-ci."""
    if not parse_yes_no(answer):
        return False
    return reject_answer is None or not parse_yes_no(reject_answer)


def clip_keep(score: float, threshold: float = config.CLIP_KEEP_THRESHOLD) -> bool:
    """Décision CLIP : image gardée si le score atteint le seuil."""
    return score >= threshold


# ---------------------------------------------------------------------------
# BLIP-2
# ---------------------------------------------------------------------------
def load_blip2(device: str = "cuda"):
    """Charge le processeur et le modèle BLIP-2 (config.BLIP2_MODEL).

    Sur GPU, le modèle est chargé en float16 (environ 8 Go de mémoire), ou en
    8 bits si config.BLIP2_LOAD_8BIT est vrai (environ 4 Go, demande la
    bibliothèque bitsandbytes). Sur CPU il est chargé en float32 : cela
    fonctionne mais demande environ 15 Go de mémoire vive et reste très lent.
    Retourne le couple (processor, model).
    """
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    log.info("Chargement de %s sur %s", config.BLIP2_MODEL, device)
    processor = Blip2Processor.from_pretrained(config.BLIP2_MODEL)
    if device == "cuda" and config.BLIP2_LOAD_8BIT:
        from transformers import BitsAndBytesConfig
        model = Blip2ForConditionalGeneration.from_pretrained(
            config.BLIP2_MODEL, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            config.BLIP2_MODEL, **dtype_kwargs(device)).to(device)
    model.eval()
    return processor, model


def _to_device(inputs, device: str, dtype) -> dict:
    """Déplace les entrées sur le matériel de calcul ; les pixels prennent la précision du modèle."""
    moved = {key: value.to(device) for key, value in inputs.items()}
    if "pixel_values" in moved:
        moved["pixel_values"] = moved["pixel_values"].to(dtype)
    return moved


def blip2_answers(image_paths: list[Path], prompt: str, processor, model,
                  device: str = "cuda",
                  batch_size: int = config.VLM_BATCH_SIZE) -> list[str]:
    """Pose la même question à BLIP-2 sur plusieurs images, par lots.

    Toutes les questions d'un lot sont identiques, donc de même longueur :
    aucun remplissage (padding) n'est nécessaire. Retourne les réponses
    nettoyées, dans l'ordre des images.
    """
    import torch
    from tqdm import tqdm

    answers: list[str] = []
    for batch in tqdm(batched(image_paths, batch_size), desc="BLIP-2", unit="lot", leave=False):
        images = [load_rgb(path) for path in batch]
        inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt")
        inputs = _to_device(inputs, device, model.dtype)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=config.BLIP2_MAX_NEW_TOKENS)
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        answers.extend(clean_answer(text) for text in decoded)
    return answers


def blip2_answer(image_path: Path, prompt: str, processor, model, device: str = "cuda") -> str:
    """Pose la question prompt à BLIP-2 sur une image et retourne la réponse nettoyée."""
    return blip2_answers([image_path], prompt, processor, model, device, batch_size=1)[0]


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------
def load_clip(device: str = "cuda"):
    """Charge le processeur et le modèle CLIP (config.CLIP_MODEL).

    Retourne le couple (processor, model).
    """
    from transformers import CLIPModel, CLIPProcessor

    log.info("Chargement de %s sur %s", config.CLIP_MODEL, device)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    model = CLIPModel.from_pretrained(config.CLIP_MODEL, **dtype_kwargs(device)).to(device)
    model.eval()
    return processor, model


def clip_probabilities(images: list, texts: list[str], processor, model,
                       device: str = "cuda") -> list[list[float]]:
    """Probabilités de chaque description pour chaque image (une ligne par image).

    Les probabilités d'une image somment à 1 sur l'ensemble des descriptions.
    Fonction générique, réutilisable par parked.py sur les recadrages de véhicules.
    """
    import torch

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = _to_device(inputs, device, model.dtype)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
    return logits.float().softmax(dim=-1).cpu().tolist()


def clip_scores(image_paths: list[Path], processor, model, device: str = "cuda",
                batch_size: int = config.VLM_BATCH_SIZE) -> list[float]:
    """Score CLIP de chaque image : somme des probabilités des descriptions positives.

    Les descriptions sont config.CLIP_POSITIVE_PROMPTS et config.CLIP_NEGATIVE_PROMPTS.
    """
    from tqdm import tqdm

    texts = config.CLIP_POSITIVE_PROMPTS + config.CLIP_NEGATIVE_PROMPTS
    n_positive = len(config.CLIP_POSITIVE_PROMPTS)
    scores: list[float] = []
    for batch in tqdm(batched(image_paths, batch_size), desc="CLIP", unit="lot", leave=False):
        images = [load_rgb(path) for path in batch]
        for probs in clip_probabilities(images, texts, processor, model, device):
            scores.append(round(sum(probs[:n_positive]), 4))
    return scores


def clip_score(image_path: Path, processor, model, device: str = "cuda") -> float:
    """Score CLIP d'une seule image (voir clip_scores)."""
    return clip_scores([image_path], processor, model, device, batch_size=1)[0]


# ---------------------------------------------------------------------------
# Étape 2
# ---------------------------------------------------------------------------
def selected_images(zone_dir: Path) -> list[tuple[str, Path]]:
    """(image_id, chemin) des images téléchargées et retenues dans metadata.csv."""
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
                images.append((row["image_id"], path))
            else:
                log.warning("Image absente du disque : %s", path)
    return images


def _write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    """Écrit un CSV avec exactement les colonnes données."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def filter_images(zone_dir: Path, method: str = config.DEFAULT_FILTER,
                  device: str = "cuda") -> Path:
    """Exécute l'étape 2 sur les images sélectionnées de metadata.csv.

    method vaut "blip2", "clip" ou "none" (toutes les images sont gardées).
    Pour BLIP-2, la question utilisée est config.BLIP2_PROMPTS[config.BLIP2_PROMPT_INDEX],
    complétée par la question de rejet si config.BLIP2_USE_REJECT est vrai.
    Écrit filter.csv dans zone_dir et retourne son chemin.
    """
    if method not in FILTER_METHODS:
        raise ValueError(f"méthode inconnue : {method!r} (choix : {', '.join(FILTER_METHODS)})")
    zone_dir = Path(zone_dir)
    images = selected_images(zone_dir)
    ids = [image_id for image_id, _ in images]
    paths = [path for _, path in images]

    if method == "none":
        rows = [{"image_id": i, "method": "none", "answer": "", "score": "", "keep": True}
                for i in ids]
    elif method == "blip2":
        device = resolve_device(device)
        processor, model = load_blip2(device)
        prompt = config.BLIP2_PROMPTS[config.BLIP2_PROMPT_INDEX]
        answers = blip2_answers(paths, prompt, processor, model, device)
        rejects: list[str | None] = [None] * len(ids)
        if config.BLIP2_USE_REJECT:
            rejects = blip2_answers(paths, config.BLIP2_REJECT_PROMPT, processor, model, device)
        del model
        free_gpu_memory()
        rows = [{"image_id": i, "method": "blip2",
                 "answer": a if r is None else f"{a} | parking : {r}",
                 "score": "", "keep": blip2_keep(a, r)}
                for i, a, r in zip(ids, answers, rejects)]
    else:
        device = resolve_device(device)
        processor, model = load_clip(device)
        scores = clip_scores(paths, processor, model, device)
        del model
        free_gpu_memory()
        rows = [{"image_id": i, "method": "clip", "answer": "", "score": s, "keep": clip_keep(s)}
                for i, s in zip(ids, scores)]

    output = zone_dir / config.FILTER_FILE
    _write_csv(rows, config.FILTER_COLUMNS, output)
    n_keep = sum(bool(r["keep"]) for r in rows)
    log.info("Filtre %s : %d image(s) gardée(s) sur %d", method, n_keep, len(rows))
    return output


def compute_variants(zone_dir: Path, device: str = "cuda",
                     methods: tuple[str, ...] = ("blip2", "clip")) -> Path:
    """Calcule toutes les variantes du filtre pour chaque image, dans filter_variants.csv.

    Colonnes : image_id, réponse à chaque question de config.BLIP2_PROMPTS
    (blip2_q0, blip2_q1...), réponse à la question de rejet, score CLIP.
    Les modèles sont chargés l'un après l'autre pour tenir dans la mémoire du GPU.
    Ce fichier sert à choisir la question, la règle et le seuil sur la partie
    « réglage » du jeu annoté, sans relancer les modèles à chaque essai.
    """
    zone_dir = Path(zone_dir)
    device = resolve_device(device)
    images = selected_images(zone_dir)
    paths = [path for _, path in images]
    rows = [{"image_id": image_id} for image_id, _ in images]

    if "blip2" in methods:
        processor, model = load_blip2(device)
        prompts = config.BLIP2_PROMPTS + [config.BLIP2_REJECT_PROMPT]
        columns = VARIANT_COLUMNS[1:len(config.BLIP2_PROMPTS) + 2]
        for prompt, column in zip(prompts, columns):
            log.info("BLIP-2, question %s : %s", column, prompt)
            for row, answer in zip(rows, blip2_answers(paths, prompt, processor, model, device)):
                row[column] = answer
        del model
        free_gpu_memory()

    if "clip" in methods:
        processor, model = load_clip(device)
        for row, score in zip(rows, clip_scores(paths, processor, model, device)):
            row["clip_score"] = score
        del model
        free_gpu_memory()

    output = zone_dir / config.FILTER_VARIANTS_FILE
    _write_csv([{c: row.get(c, "") for c in VARIANT_COLUMNS} for row in rows],
               VARIANT_COLUMNS, output)
    log.info("Variantes du filtre pour %d image(s) dans %s", len(rows), output)
    return output


def main(argv: list[str] | None = None) -> int:
    """Lance le filtre seul sur une zone déjà collectée, ou calcule ses variantes."""
    parser = argparse.ArgumentParser(description="Filtrage des images d'une zone (étape 2).")
    parser.add_argument("--zone", type=Path, required=True,
                        help="dossier de la zone, par exemple data/carrefour_pistes")
    parser.add_argument("--method", choices=FILTER_METHODS, default=config.DEFAULT_FILTER,
                        help="méthode de filtrage (défaut : %(default)s)")
    parser.add_argument("--variants", action="store_true",
                        help="calculer toutes les variantes dans filter_variants.csv "
                             "au lieu d'écrire filter.csv")
    parser.add_argument("--only", choices=("blip2", "clip"), default=None,
                        help="avec --variants : un seul des deux modèles")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")

    try:
        if args.variants:
            methods = (args.only,) if args.only else ("blip2", "clip")
            compute_variants(args.zone, args.device, methods)
        else:
            filter_images(args.zone, args.method, args.device)
    except RuntimeError as error:
        log.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

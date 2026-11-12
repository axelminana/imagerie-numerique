"""Tests du filtre (étape 2), sans télécharger de modèle.

Les règles de décision sont testées directement. Pour CLIP, on construit un
petit modèle CLIP aléatoire à partir d'une configuration (aucun téléchargement) ;
pour BLIP-2, un faux modèle imite generate(). Lancer :
    python -m pytest tests/test_filtering.py -v
"""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import filtering  # noqa: E402


# ---------------------------------------------------------------------------
# Règles de décision
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decoded,expected", [
    ("yes", "yes"),
    ("Question: Is this a street? Answer yes or no. Answer: no", "no"),
    ("  Yes, it is\nsomething else", "Yes, it is"),
    ("", ""),
])
def test_clean_answer(decoded, expected):
    assert filtering.clean_answer(decoded) == expected


@pytest.mark.parametrize("answer,reject,expected", [
    ("yes", None, True), ("no", None, False),
    ("yes", "no", True), ("yes", "yes", False), ("no", "no", False),
])
def test_blip2_keep(answer, reject, expected):
    assert filtering.blip2_keep(answer, reject) is expected


def test_clip_keep():
    assert filtering.clip_keep(0.7, threshold=0.5)
    assert filtering.clip_keep(0.5, threshold=0.5)
    assert not filtering.clip_keep(0.3, threshold=0.5)


# ---------------------------------------------------------------------------
# Zone factice
# ---------------------------------------------------------------------------
def make_zone(root, n=3):
    """Zone avec n images RGB réelles (petites) et un metadata.csv."""
    from PIL import Image

    zone = root / "essai"
    (zone / "images").mkdir(parents=True)
    with open(zone / config.METADATA_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.METADATA_COLUMNS)
        writer.writeheader()
        for i in range(1, n + 1):
            Image.new("RGB", (64, 48), (40 * i, 80, 120)).save(zone / "images" / f"{i}.jpg")
            writer.writerow({"image_id": str(i), "sequence_id": "s", "lat": 45.8, "lon": 3.1,
                             "distance_m": 10, "captured_at": "", "compass_angle": "",
                             "is_pano": False, "selected": True, "path": f"images/{i}.jpg"})
        writer.writerow({"image_id": "99", "sequence_id": "s", "lat": 45.8, "lon": 3.1,
                         "distance_m": 150, "captured_at": "", "compass_angle": "",
                         "is_pano": "", "selected": False, "path": ""})
    return zone


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_filtre_none(tmp_path):
    zone = make_zone(tmp_path)
    rows = read_rows(filtering.filter_images(zone, "none"))
    assert [r["image_id"] for r in rows] == ["1", "2", "3"]
    assert all(r["keep"] == "True" for r in rows)
    assert list(rows[0]) == config.FILTER_COLUMNS


def test_methode_inconnue(tmp_path):
    with pytest.raises(ValueError):
        filtering.filter_images(make_zone(tmp_path), "yolo")


# ---------------------------------------------------------------------------
# CLIP : petit modèle aléatoire, vraie bibliothèque transformers
# ---------------------------------------------------------------------------
class FakeClipProcessor:
    """Remplace CLIPProcessor : convertit les images en tenseurs 32 x 32 et les textes en jetons."""

    def __call__(self, text, images, return_tensors, padding):
        import numpy as np
        import torch

        pixels = np.stack([np.asarray(img.resize((32, 32)), dtype="float32") / 255
                           for img in images]).transpose(0, 3, 1, 2)
        ids = torch.tensor([[1 + (hash(t) + k) % 90 for k in range(6)] for t in text])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.from_numpy(pixels)}


@pytest.fixture
def tiny_clip(monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import torch

    torch.manual_seed(0)
    model_config = transformers.CLIPConfig(
        text_config={"vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 2,
                     "max_position_embeddings": 16},
        vision_config={"image_size": 32, "patch_size": 16, "hidden_size": 16,
                       "intermediate_size": 32, "num_hidden_layers": 1,
                       "num_attention_heads": 2},
        projection_dim=8)
    model = transformers.CLIPModel(model_config).eval()
    processor = FakeClipProcessor()
    monkeypatch.setattr(filtering, "load_clip", lambda device: (processor, model))
    monkeypatch.setattr(filtering, "resolve_device", lambda device: "cpu")
    return processor, model


def test_clip_probabilities_somment_a_un(tiny_clip, tmp_path):
    from PIL import Image

    processor, model = tiny_clip
    images = [Image.new("RGB", (40, 30), c) for c in ((255, 0, 0), (0, 255, 0))]
    probs = filtering.clip_probabilities(images, ["a", "b", "c"], processor, model, "cpu")
    assert len(probs) == 2 and all(len(p) == 3 for p in probs)
    assert all(sum(p) == pytest.approx(1, abs=1e-5) for p in probs)


def test_filtre_clip(tiny_clip, tmp_path, monkeypatch):
    zone = make_zone(tmp_path, n=5)
    monkeypatch.setattr(config, "VLM_BATCH_SIZE", 2)          # plusieurs lots
    monkeypatch.setattr(filtering, "clip_keep", lambda s: s >= 0.5)
    rows = read_rows(filtering.filter_images(zone, "clip", device="cpu"))

    assert [r["image_id"] for r in rows] == ["1", "2", "3", "4", "5"]
    for row in rows:
        score = float(row["score"])
        assert 0 <= score <= 1
        assert row["method"] == "clip"
        assert row["keep"] == str(score >= 0.5)


# ---------------------------------------------------------------------------
# BLIP-2 : faux modèle qui imite generate() et batch_decode()
# ---------------------------------------------------------------------------
class FakeBlip2:
    """Imite generate() : transmet la question et la couleur ; FakeBlip2Processor décide."""

    dtype = "float16"

    def __init__(self):
        self.prompts = []

    def generate(self, pixel_values, input_ids, max_new_tokens):
        return [(prompt, red) for prompt, red in zip(input_ids, pixel_values)]


class FakeTensor:
    """Valeur quelconque avec une méthode .to() qui la laisse inchangée."""

    def __init__(self, values):
        self.values = values

    def to(self, *args):
        return self

    def __iter__(self):
        return iter(self.values)


class FakeBlip2Processor:
    """Transmet la question et la couleur rouge de chaque image au faux modèle."""

    def __call__(self, images, text, return_tensors):
        return {"pixel_values": FakeTensor([img.getpixel((0, 0))[0] for img in images]),
                "input_ids": FakeTensor(text)}

    def batch_decode(self, generated, skip_special_tokens):
        answers = []
        for prompt, red in generated:
            if "parking" in prompt:
                answer = "yes" if 60 <= red < 100 else "no"     # image 2 (rouge ~80) : parking
            else:
                answer = "yes" if red < 100 else "no"           # images 1 et 2 : rue
            # comme certaines versions de transformers, on renvoie la question avec la réponse
            answers.append(f"{prompt} {answer}")
        return answers


@pytest.fixture
def fake_blip2(monkeypatch):
    pytest.importorskip("torch")
    model = FakeBlip2()
    monkeypatch.setattr(filtering, "load_blip2", lambda device: (FakeBlip2Processor(), model))
    monkeypatch.setattr(filtering, "resolve_device", lambda device: "cpu")
    return model


def test_filtre_blip2(fake_blip2, tmp_path, monkeypatch):
    zone = make_zone(tmp_path)                       # rouge : 40, 80, 120
    rows = read_rows(filtering.filter_images(zone, "blip2", device="cpu"))
    assert [(r["answer"], r["keep"]) for r in rows] == [
        ("yes", "True"), ("yes", "True"), ("no", "False")]


def test_filtre_blip2_avec_question_de_rejet(fake_blip2, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BLIP2_USE_REJECT", True)
    zone = make_zone(tmp_path)
    rows = read_rows(filtering.filter_images(zone, "blip2", device="cpu"))
    assert [r["keep"] for r in rows] == ["True", "False", "False"]
    assert rows[1]["answer"] == "yes | parking : yes"


def test_variantes(fake_blip2, tiny_clip, tmp_path):
    zone = make_zone(tmp_path)
    rows = read_rows(filtering.compute_variants(zone, device="cpu"))
    assert list(rows[0]) == filtering.VARIANT_COLUMNS
    assert [r["blip2_q0"] for r in rows] == ["yes", "yes", "no"]
    assert [r["blip2_reject"] for r in rows] == ["no", "yes", "no"]
    assert all(0 <= float(r["clip_score"]) <= 1 for r in rows)


def test_variantes_clip_seul(tiny_clip, tmp_path, monkeypatch):
    monkeypatch.setattr(filtering, "load_blip2", lambda d: pytest.fail("BLIP-2 non demandé"))
    rows = read_rows(filtering.compute_variants(make_zone(tmp_path), "cpu", methods=("clip",)))
    assert all(r["blip2_q0"] == "" and r["clip_score"] for r in rows)


def test_main_message_clair_sans_metadata(tmp_path):
    assert filtering.main(["--zone", str(tmp_path / "vide"), "--method", "none"]) == 1


def test_dtype_kwargs_selon_la_version(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from utils import dtype_kwargs

    monkeypatch.setattr(transformers, "__version__", "4.44.2")
    assert dtype_kwargs("cuda") == {"torch_dtype": torch.float16}
    monkeypatch.setattr(transformers, "__version__", "5.1.0")
    assert dtype_kwargs("cpu") == {"dtype": torch.float32}

# Découpage train / val / test (70 / 15 / 15, stratifié par classe),
# puis redimensionnement en 640x640 et égalisation CLAHE.
#
# Le découpage se fait sur les images d'origine, avant toute augmentation :
# aucune image de val ou de test n'a de copie augmentée dans le train.

import json
import random
import sys

import cv2
from tqdm import tqdm

from config import CLASSES, ORIGINAL, PROCESSED, lister_images

TAILLE = (640, 640)
RATIO_TRAIN, RATIO_VAL = 0.70, 0.15
SEED = 42


def clahe(image, clip=2.0, grille=(8, 8)):
    # On égalise seulement la luminance (canal L), pour ne pas toucher aux couleurs
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=grille).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def pretraiter(chemin):
    image = cv2.cvtColor(cv2.imread(str(chemin)), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, TAILLE, interpolation=cv2.INTER_AREA)
    return clahe(image)


def decouper():
    random.seed(SEED)
    splits = {"train": {}, "val": {}, "test": {}}
    stats = {}

    for c in CLASSES:
        fichiers = lister_images(ORIGINAL / c)
        random.shuffle(fichiers)
        n_train = int(len(fichiers) * RATIO_TRAIN)
        n_val = int(len(fichiers) * RATIO_VAL)

        splits["train"][c] = fichiers[:n_train]
        splits["val"][c] = fichiers[n_train:n_train + n_val]
        splits["test"][c] = fichiers[n_train + n_val:]
        stats[c] = {s: len(splits[s][c]) for s in splits}
        print(f"{c:50s} {stats[c]['train']:5d} train {stats[c]['val']:4d} val {stats[c]['test']:4d} test")

    return splits, stats


def main():
    if PROCESSED.exists():
        sys.exit(f"{PROCESSED} existe déjà : supprimez-le pour refaire le prétraitement.")

    splits, stats = decouper()
    total = sum(len(f) for s in splits.values() for f in s.values())

    with tqdm(total=total, unit="img") as barre:
        for split, par_classe in splits.items():
            for c, fichiers in par_classe.items():
                dossier = PROCESSED / split / c
                dossier.mkdir(parents=True, exist_ok=True)
                for chemin in fichiers:
                    image = pretraiter(chemin)
                    cv2.imwrite(str(dossier / chemin.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    barre.update()

    with open(PROCESSED / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump({"taille": TAILLE, "seed": SEED, "stats": stats}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

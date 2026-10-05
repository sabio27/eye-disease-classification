# Augmentation du train uniquement, plus forte pour les classes rares.
# Val et test sont copiés tels quels.

import shutil
import sys

import albumentations as A
import cv2
from tqdm import tqdm

from config import AUGMENTED, PROCESSED, lister_images

# Nombre de versions par image d'origine (1 = pas d'augmentation).
# Les commentaires donnent le nombre d'images de train avant -> après.
FACTEURS = {
    "Diabetic Retinopathy": 1,                               # 1056
    "Glaucoma": 1,                                           # 944
    "Healthy": 1,                                            # 716
    "Myopia": 2,                                             # 350 -> 700
    "Macular Scar": 3,                                       # 310 -> 930
    "Retinitis Pigmentosa": 8,                               # 97 -> 776
    "Disc Edema": 10,                                        # 88 -> 880
    "Retinal Detachment": 10,                                # 87 -> 870
    "Central Serous Chorioretinopathy [Color Fundus]": 12,   # 70 -> 840
    "Pterygium": 50,                                         # 11 -> 550
}

TRANSFORMATIONS = A.Compose([
    A.Rotate(limit=15, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.5),
])


def augmenter_classe(classe, facteur):
    source = PROCESSED / "train" / classe
    cible = AUGMENTED / "train" / classe
    cible.mkdir(parents=True, exist_ok=True)

    fichiers = lister_images(source)
    for chemin in tqdm(fichiers, desc=classe, leave=False):
        shutil.copy(chemin, cible / chemin.name)
        if facteur == 1:
            continue
        image = cv2.cvtColor(cv2.imread(str(chemin)), cv2.COLOR_BGR2RGB)
        for i in range(facteur - 1):
            nouvelle = TRANSFORMATIONS(image=image)["image"]
            cv2.imwrite(str(cible / f"{chemin.stem}_aug{i + 1}.jpg"),
                        cv2.cvtColor(nouvelle, cv2.COLOR_RGB2BGR))

    print(f"{classe:50s} {len(fichiers):5d} -> {len(fichiers) * facteur:5d}")


def main():
    if not PROCESSED.exists():
        sys.exit("Lancez d'abord src/preprocess.py")
    if AUGMENTED.exists():
        sys.exit(f"{AUGMENTED} existe déjà : supprimez-le pour refaire l'augmentation.")

    for split in ["val", "test"]:
        shutil.copytree(PROCESSED / split, AUGMENTED / split)
    for classe, facteur in FACTEURS.items():
        augmenter_classe(classe, facteur)


if __name__ == "__main__":
    main()

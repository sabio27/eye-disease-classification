# Recopie le dataset augmenté avec des noms de classes sans espaces,
# dans la structure attendue par YOLO en classification (un dossier par classe).

import shutil
import sys

from config import AUGMENTED, CLASSES, YOLO_DATA, lister_images


def main():
    if not AUGMENTED.exists():
        sys.exit("Lancez d'abord src/augment.py")
    if YOLO_DATA.exists():
        sys.exit(f"{YOLO_DATA} existe déjà : supprimez-le pour le recréer.")

    for split in ["train", "val", "test"]:
        n = 0
        for nom_origine, nom_yolo in CLASSES.items():
            cible = YOLO_DATA / split / nom_yolo
            cible.mkdir(parents=True, exist_ok=True)
            for chemin in lister_images(AUGMENTED / split / nom_origine):
                shutil.copy(chemin, cible / chemin.name)
                n += 1
        print(f"{split:5s} : {n} images")


if __name__ == "__main__":
    main()

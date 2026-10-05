# Sous-ensemble du train pour l'autoencodeur : les 4 classes les plus
# fournies, 800 images maximum chacune. Les classes rares restent hors de
# son entraînement, ce qui permet ensuite de voir si elles se reconstruisent
# moins bien (détection d'anomalies).

import random
import shutil

from config import AE_TRAIN, PROCESSED, lister_images

CLASSES_AE = ["Healthy", "Diabetic Retinopathy", "Glaucoma", "Myopia"]
MAX_PAR_CLASSE = 800


def main():
    for classe in CLASSES_AE:
        fichiers = lister_images(PROCESSED / "train" / classe)
        random.shuffle(fichiers)
        cible = AE_TRAIN / classe
        cible.mkdir(parents=True, exist_ok=True)
        for chemin in fichiers[:MAX_PAR_CLASSE]:
            shutil.copy2(chemin, cible / chemin.name)
        print(f"{classe:25s} {min(len(fichiers), MAX_PAR_CLASSE)} images")


if __name__ == "__main__":
    main()

# Premier coup d'oeil au dataset : nombre d'images par classe,
# taille et luminosité des images, quelques exemples.

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import CLASSES, ORIGINAL, lister_images


def compter_images():
    compte = {c: len(lister_images(ORIGINAL / c)) for c in CLASSES}
    for c, n in compte.items():
        print(f"{c:50s} {n:5d}")
    print(f"Total : {sum(compte.values())} images")

    plus, moins = max(compte.values()), min(n for n in compte.values() if n > 0)
    print(f"Déséquilibre : {plus / moins:.0f}x entre la classe la plus et la moins fournie")
    return compte


def tracer_distribution(compte):
    classes, valeurs = list(compte), list(compte.values())
    plt.figure(figsize=(10, 6))
    plt.barh(classes, valeurs, color="steelblue")
    for i, v in enumerate(valeurs):
        plt.text(v + 10, i, str(v), va="center")
    plt.xlabel("Nombre d'images")
    plt.title("Images par classe (dataset d'origine)")
    plt.tight_layout()
    plt.savefig("results/class_distribution.png", dpi=150)
    plt.close()


def proprietes_images(par_classe=50):
    tailles, poids, luminosite = [], [], []
    for c in CLASSES:
        for chemin in lister_images(ORIGINAL / c)[:par_classe]:
            img = Image.open(chemin)
            tailles.append(img.size)
            poids.append(os.path.getsize(chemin) / 1024)
            luminosite.append(np.asarray(img).mean())

    largeurs, hauteurs = zip(*tailles)
    print(f"Largeur {min(largeurs)}-{max(largeurs)} px, hauteur {min(hauteurs)}-{max(hauteurs)} px")
    print(f"Fichiers de {min(poids):.0f} à {max(poids):.0f} Ko")
    print(f"Luminosité moyenne de {min(luminosite):.0f} à {max(luminosite):.0f}")


def exemples(par_classe=2):
    fig, axes = plt.subplots(len(CLASSES), par_classe, figsize=(par_classe * 4, len(CLASSES) * 3))
    for i, c in enumerate(CLASSES):
        for j, chemin in enumerate(lister_images(ORIGINAL / c)[:par_classe]):
            axes[i, j].imshow(Image.open(chemin))
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(c, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/sample_images.png", dpi=80)
    plt.close()


if __name__ == "__main__":
    compte = compter_images()
    tracer_distribution(compte)
    proprietes_images()
    exemples()

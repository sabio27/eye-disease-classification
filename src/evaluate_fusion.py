# Compare YOLO-EyeNet, l'autoencodeur et leur fusion sur le jeu de test.
#
# Chaque jeu a son rôle :
#   - train      : apprentissage du MLP
#   - validation : choix de la meilleure époque
#   - test       : évaluation finale, une seule fois
#
# YOLO-EyeNet et l'autoencodeur ne sont pas réentraînés : on réutilise
# les poids existants. Aucune image de validation ou de test n'a servi à
# les entraîner (vérifié sur les noms de fichiers).
#
# À lancer depuis la racine du dépôt :
#     python src/evaluate_fusion.py

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from ultralytics import YOLO

from autoencoder import ConvAutoencoder
from config import RESULTS, YOLO_DATA

DATA = YOLO_DATA
# Ultralytics range les poids sous runs/classify/<project>
YOLO_POIDS = Path("runs") / "classify" / RESULTS / "YOLO_EyeNet" / "train" / "weights" / "best.pt"
AE_POIDS = RESULTS / "autoencoder_eye.pth"
SORTIE = Path("results")
CACHE = RESULTS / "embeddings.npz"   # gros fichier, hors du dépôt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def charger(split):
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ds = datasets.ImageFolder(DATA / split, transform=tf)
    return ds, DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)


def extraire(split, yolo, ae, yolo_idx):
    """Renvoie embeddings YOLO, embeddings AE, probas YOLO et labels d'un split."""
    ds, loader = charger(split)

    # L'embedding YOLO est l'entrée de la dernière couche linéaire de la tête
    # de classification (vecteur après convolution et pooling).
    captures = []
    hook = yolo.model.model[-1].linear.register_forward_hook(
        lambda m, inp, out: captures.append(inp[0].detach().cpu()))

    emb_yolo, emb_ae, probas, labels = [], [], [], []
    with torch.no_grad():
        for i, (imgs, y) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            sortie = yolo.model(imgs)
            if isinstance(sortie, (tuple, list)):
                sortie = sortie[0]
            # remet les colonnes dans l'ordre des classes d'ImageFolder
            probas.append(sortie[:, yolo_idx].cpu())
            emb_ae.append(ae.encode(imgs).cpu())
            labels.append(y)
            if i % 20 == 0:
                print(f"  {split} : {i * 32}/{len(ds)}")

    hook.remove()
    return (torch.cat(captures).numpy(), torch.cat(emb_ae).numpy(),
            torch.cat(probas).numpy(), torch.cat(labels).numpy(), ds.classes)


class FusionMLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def entrainer_mlp(X_tr, y_tr, X_val, y_val, n_classes, epochs=60):
    """Entraîne le MLP et garde l'époque au meilleur F1 macro en validation."""
    torch.manual_seed(SEED)

    # standardisation apprise sur le train uniquement
    moy, ecart = X_tr.mean(0), X_tr.std(0) + 1e-6
    norm = lambda X: torch.tensor((X - moy) / ecart, dtype=torch.float32)

    loader = DataLoader(TensorDataset(norm(X_tr), torch.tensor(y_tr)),
                        batch_size=64, shuffle=True)
    Xv = norm(X_val)

    model = FusionMLP(X_tr.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    perte = nn.CrossEntropyLoss()

    meilleur, meilleur_etat, meilleure_epoque = -1, None, 0
    for epoque in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            perte(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(1).numpy()
        f1 = f1_score(y_val, pred, average="macro")
        if f1 > meilleur:
            meilleur, meilleure_epoque = f1, epoque + 1
            meilleur_etat = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(meilleur_etat)
    model.eval()
    predire = lambda X: model(norm(X)).argmax(1).detach().numpy()
    return predire, meilleure_epoque, meilleur, model


def resume(nom, y, pred):
    return {"modele": nom,
            "accuracy": round(accuracy_score(y, pred) * 100, 2),
            "f1_macro": round(f1_score(y, pred, average="macro"), 4)}


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("Chargement des modèles...")
    yolo = YOLO(str(YOLO_POIDS))
    yolo.model.to(DEVICE).eval()

    ae = ConvAutoencoder(latent_dim=512, num_classes=4)
    ae.load_state_dict(torch.load(AE_POIDS, map_location=DEVICE))
    ae.to(DEVICE).eval()

    classes = datasets.ImageFolder(DATA / "test").classes
    noms_yolo = {v: k for k, v in yolo.names.items()}
    yolo_idx = [noms_yolo[c] for c in classes]

    cache = CACHE
    if cache.exists():
        print("Embeddings déjà extraits, on les recharge.")
        d = np.load(cache)
        E = {k: d[k] for k in d.files}
    else:
        print("Extraction des embeddings (quelques minutes sur CPU)...")
        E = {}
        for split in ["train", "val", "test"]:
            ey, ea, p, y, _ = extraire(split, yolo, ae, yolo_idx)
            E.update({f"{split}_yolo": ey, f"{split}_ae": ea,
                      f"{split}_proba": p, f"{split}_y": y})
        np.savez_compressed(cache, **E)

    n = len(classes)
    y_test = E["test_y"]
    lignes, rapports = [], {}

    # 1. YOLO-EyeNet seul, tel qu'entraîné
    pred_yolo = E["test_proba"].argmax(1)
    lignes.append(resume("YOLO-EyeNet seul", y_test, pred_yolo))
    rapports["YOLO-EyeNet seul"] = pred_yolo

    # 2. MLP sur chaque type d'embedding, puis sur la fusion
    variantes = {
        "MLP sur embeddings YOLO": ["yolo"],
        "MLP sur embeddings autoencodeur": ["ae"],
        "Fusion YOLO + autoencodeur": ["yolo", "ae"],
    }
    for nom, sources in variantes.items():
        X = {s: np.concatenate([E[f"{s}_{src}"] for src in sources], axis=1)
             for s in ["train", "val", "test"]}
        predire, epoque, f1_val, model = entrainer_mlp(
            X["train"], E["train_y"], X["val"], E["val_y"], n)
        pred = predire(X["test"])
        ligne = resume(nom, y_test, pred)
        ligne.update({"meilleure_epoque": epoque, "f1_macro_val": round(f1_val, 4)})
        lignes.append(ligne)
        rapports[nom] = pred
        if sources == ["yolo", "ae"]:
            torch.save(model.state_dict(), RESULTS / "fusion_mlp.pth")

    print("\nRésultats sur le jeu de test (810 images)")
    for l in lignes:
        print(l)

    with open(SORTIE / "resultats.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "resultats": lignes,
                   "rapports": {k: classification_report(y_test, v, target_names=classes,
                                                         output_dict=True, zero_division=0)
                                for k, v in rapports.items()}},
                  f, indent=2, ensure_ascii=False)

    for nom in ["YOLO-EyeNet seul", "Fusion YOLO + autoencodeur"]:
        print(f"\n=== {nom} ===")
        print(classification_report(y_test, rapports[nom], target_names=classes,
                                    digits=3, zero_division=0))

        cm = confusion_matrix(y_test, rapports[nom])
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.title(f"Matrice de confusion (test) - {nom}")
        plt.tight_layout()
        fichier = "confusion_yolo.png" if "YOLO-EyeNet" in nom else "confusion_fusion.png"
        plt.savefig(SORTIE / fichier, dpi=120)
        plt.close()

    print(f"\nTout est enregistré dans {SORTIE}")


if __name__ == "__main__":
    main()

# Chemins et classes communs à tous les scripts.
# Les scripts se lancent depuis la racine du dépôt : python src/<script>.py

from pathlib import Path

DATASET = Path("Eye Disease Image Dataset")
ORIGINAL = DATASET / "Original Dataset"        # images téléchargées
PROCESSED = DATASET / "Processed Dataset"      # redimensionnées + CLAHE, découpées
AUGMENTED = DATASET / "Augmented Dataset"      # train augmenté, val/test inchangés
YOLO_DATA = DATASET / "YOLO Dataset"           # noms de dossiers sans espaces
AE_TRAIN = DATASET / "processed" / "autoencoder" / "train"
RESULTS = DATASET / "Results"

# Nom du dossier d'origine -> nom utilisé pour YOLO
CLASSES = {
    "Diabetic Retinopathy": "Diabetic_Retinopathy",
    "Glaucoma": "Glaucoma",
    "Macular Scar": "Macular_Scar",
    "Disc Edema": "Disc_Edema",
    "Central Serous Chorioretinopathy [Color Fundus]": "Central_Serous_Chorioretinopathy",
    "Retinal Detachment": "Retinal_Detachment",
    "Retinitis Pigmentosa": "Retinitis_Pigmentosa",
    "Myopia": "Myopia",
    "Pterygium": "Pterygium",
    "Healthy": "Healthy",
}

IMAGES = ("*.jpg", "*.jpeg", "*.png")


def lister_images(dossier):
    return [f for motif in IMAGES for f in Path(dossier).glob(motif)]

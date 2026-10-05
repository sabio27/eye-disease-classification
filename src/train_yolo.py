# Fine-tuning des modèles YOLO de classification.
#
#   python src/train_yolo.py yolov8s   -> YOLOv8s-cls, réglages par défaut
#   python src/train_yolo.py yolo11s   -> YOLO11s-cls, réglages par défaut
#   python src/train_yolo.py eyenet    -> "YOLO-EyeNet" : YOLO11s-cls entraîné plus
#                                         longtemps, LR plus faible, AdamW, dropout

import sys
from datetime import datetime

import torch
from ultralytics import YOLO

from config import RESULTS, YOLO_DATA

REGLAGES = {
    "yolov8s": dict(poids="yolov8s-cls.pt", dossier="YOLOv8s", epochs=50, patience=10),
    "yolo11s": dict(poids="yolo11s-cls.pt", dossier="YOLOv11s", epochs=50, patience=10),
    "eyenet": dict(poids="yolo11s-cls.pt", dossier="YOLO_EyeNet", epochs=80, patience=15,
                   lr0=5e-3, optimizer="AdamW", dropout=0.1),
}


def main(nom):
    r = dict(REGLAGES[nom])
    poids, dossier = r.pop("poids"), r.pop("dossier")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(poids)
    debut = datetime.now()
    model.train(data=str(YOLO_DATA), batch=16, imgsz=224, device=device,
                project=str(RESULTS / dossier), name="train", exist_ok=True, **r)
    print(f"Entraînement : {(datetime.now() - debut).total_seconds() / 3600:.2f} h")

    # Évaluation du meilleur modèle (choisi sur la validation) sur le test
    meilleur = YOLO(str(model.trainer.best))
    m = meilleur.val(data=str(YOLO_DATA), split="test")
    print(f"{nom} - test : top-1 {m.top1 * 100:.2f} %, top-5 {m.top5 * 100:.2f} %")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in REGLAGES:
        sys.exit(f"Usage : python src/train_yolo.py [{'|'.join(REGLAGES)}]")
    main(sys.argv[1])

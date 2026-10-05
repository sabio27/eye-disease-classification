# Autoencodeur convolutionnel avec une petite tête de classification sur
# l'espace latent. La perte combine reconstruction (MSE + SSIM) et
# classification (poids 0.1), pour que le latent garde une information utile
# aux classes et pas seulement à la reconstruction.
#
# Entraînement : python src/autoencoder.py

import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import AE_TRAIN, RESULTS


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=512, num_classes=4):
        super().__init__()

        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        canaux = [3, 32, 64, 128, 256, 512]
        couches = []
        for c_in, c_out in zip(canaux, canaux[1:]):
            couches += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*couches)

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(512 * 7 * 7, latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self.fc_dec = nn.Linear(latent_dim, 512 * 7 * 7)

        # Décodeur symétrique, 7 -> 224
        couches = []
        for c_in, c_out in zip(canaux[::-1], canaux[::-1][1:]):
            couches += [nn.ConvTranspose2d(c_in, c_out, 3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(inplace=True)]
        couches[-1] = nn.Sigmoid()   # sortie dans [0, 1] comme les images
        self.decoder = nn.Sequential(*couches)

    def encode(self, x):
        return self.fc_enc(self.flatten(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.fc_dec(z).view(-1, 512, 7, 7))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z, self.classifier(z)


def perte_reconstruction(rec, orig, alpha=0.5):
    return alpha * nn.functional.mse_loss(rec, orig) + (1 - alpha) * (1 - ssim(rec, orig, data_range=1.0))


def main(epochs=80, lambda_cls=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(AE_TRAIN, transform=tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = ConvAutoencoder(latent_dim=512, num_classes=len(dataset.classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    perte_cls = nn.CrossEntropyLoss()

    for epoque in range(epochs):
        model.train()
        total, justes, cumul = 0, 0, 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            rec, _, logits = model(imgs)
            perte = perte_reconstruction(rec, imgs) + lambda_cls * perte_cls(logits, labels)
            perte.backward()
            opt.step()

            cumul += perte.item() * imgs.size(0)
            justes += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        scheduler.step(cumul / total)
        print(f"Époque {epoque + 1:3d}/{epochs}  perte {cumul / total:.4f}  acc train {justes / total:.1%}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RESULTS / "autoencoder_eye.pth")


if __name__ == "__main__":
    main()

"""
train_gender_classifier.py
-----------------------------
PHASE 2 (not needed to run the baseline demo).

Once you've logged real (product_image -> gender_label) pairs from your
DB (or corrected CLIP's zero-shot guesses over time), fine-tune a small
CNN here instead of relying on CLIP zero-shot. This is a light job -
comfortably trainable on a 4-16GB GPU in Colab or locally.

Expected data layout:
    data/gender/male/*.jpg
    data/gender/female/*.jpg

Run:
    python train_gender_classifier.py --data_dir data/gender --epochs 10
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_model(num_classes=2):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Classes: {dataset.classes} (index order matters -> update gender.py mapping if different)")

    model = build_model(num_classes=len(dataset.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        print(f"Epoch {epoch+1}/{args.epochs} - loss: {total_loss/total:.4f} - acc: {correct/total:.4f}")

    torch.save(model.state_dict(), args.out_path)
    print(f"Saved model -> {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/gender")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_path", default="models/gender_classifier.pt")
    args = parser.parse_args()
    main(args)

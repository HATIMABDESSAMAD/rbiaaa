import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}


class PlantDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples(dataset_dir, metadata_csv=None):
    if metadata_csv and Path(metadata_csv).exists():
        df = pd.read_csv(metadata_csv)
        if "usable_for_training" in df.columns:
            df = df[df["usable_for_training"].astype(str).str.lower().isin(["true", "1"])]
        classes = sorted(df["class_name"].unique().tolist(), key=str.lower)
        class_to_idx = {name: i for i, name in enumerate(classes)}
        samples_by_class = {name: [] for name in classes}
        for _, row in df.iterrows():
            samples_by_class[row["class_name"]].append(Path(row["image_path"]))
        return classes, class_to_idx, samples_by_class

    dataset_dir = Path(dataset_dir)
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    classes = [p.name for p in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    samples_by_class = {name: [] for name in classes}

    for class_dir in class_dirs:
        for path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples_by_class[class_dir.name].append(path)

    return classes, class_to_idx, samples_by_class


def make_split(samples_by_class, class_to_idx, val_per_class, seed):
    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for class_name, paths in samples_by_class.items():
        paths = list(paths)
        rng.shuffle(paths)
        n_val = min(val_per_class, max(1, len(paths) - 1))
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]

        label = class_to_idx[class_name]
        train_samples.extend((str(path), label) for path in train_paths)
        val_samples.extend((str(path), label) for path in val_paths)

    return train_samples, val_samples


def build_transforms(img_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.08)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, val_transform


def make_sampler(train_samples, num_classes):
    labels = [label for _, label in train_samples]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.array([1.0 / max(counts[label], 1.0) for label in labels], dtype=np.float32)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True), counts


def topk_metrics(y_true, logits):
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    top1 = accuracy_score(y_true, logits.argmax(axis=1))
    top5 = top_k_accuracy_score(y_true, logits, k=min(5, logits.shape[1]), labels=np.arange(logits.shape[1]))
    macro_f1 = f1_score(y_true, logits.argmax(axis=1), average="macro", zero_division=0)
    return float(top1), float(top5), float(macro_f1)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    all_labels = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_logits.append(logits.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    top1, top5, macro_f1 = topk_metrics(all_labels, logits_np)
    return {
        "loss": total_loss / len(loader.dataset),
        "top1": top1,
        "top5": top5,
        "macro_f1": macro_f1,
        "labels": np.asarray(all_labels),
        "logits": logits_np,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Bases de données photos Dr Taleb")
    parser.add_argument("--metadata", default="outputs/metadata_clean.csv")
    parser.add_argument("--output", default="outputs_resnet50v2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    classes, class_to_idx, samples_by_class = collect_samples(args.dataset, args.metadata)
    train_samples, val_samples = make_split(samples_by_class, class_to_idx, args.val_per_class, args.seed)
    num_classes = len(classes)

    train_transform, val_transform = build_transforms(args.img_size)
    train_dataset = PlantDataset(train_samples, train_transform)
    val_dataset = PlantDataset(val_samples, val_transform)
    sampler, train_counts = make_sampler(train_samples, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("resnetv2_50", pretrained=True, num_classes=num_classes)
    model.to(device)

    class_weights = 1.0 / np.maximum(train_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    split_info = {
        "model": "resnetv2_50",
        "classes": num_classes,
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "val_percentage": len(val_samples) / (len(train_samples) + len(val_samples)),
        "val_per_class": args.val_per_class,
        "device": device,
        "class_to_idx": class_to_idx,
    }
    with open(output / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    with open(output / "classes.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name"])
        for class_name, class_id in class_to_idx.items():
            writer.writerow([class_id, class_name])

    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.freeze_backbone_epochs > 0:
            for name, param in model.named_parameters():
                param.requires_grad = "head" in name or "fc" in name or "classifier" in name
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        if epoch == args.freeze_backbone_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch + 1))

        train_result = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_result = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_top1": train_result["top1"],
            "train_top5": train_result["top5"],
            "train_macro_f1": train_result["macro_f1"],
            "val_loss": val_result["loss"],
            "val_top1": val_result["top1"],
            "val_top5": val_result["top5"],
            "val_macro_f1": val_result["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with open(output / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        with open(output / "history.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if val_result["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_result["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "args": vars(args),
                    "metrics": row,
                },
                output / "best_resnet50v2.pt",
            )
            y_true = val_result["labels"]
            y_pred = val_result["logits"].argmax(axis=1)
            report = classification_report(y_true, y_pred, labels=np.arange(num_classes), target_names=classes, zero_division=0, output_dict=True)
            with open(output / "best_classification_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Best validation macro F1: {best_macro_f1:.4f}")
    print(f"Best checkpoint: {output / 'best_resnet50v2.pt'}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm import tqdm

from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


class ImageFolderWithPaths(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, is_valid_file=self._is_valid_file)

    @staticmethod
    def _is_valid_file(path):
        return Path(path).suffix.lower() in IMAGE_EXTENSIONS

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            image = self.loader(path)
        except Exception:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target, path


class IndexSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        old_transform = self.dataset.transform
        self.dataset.transform = self.transform
        try:
            return self.dataset[index]
        finally:
            self.dataset.transform = old_transform


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_one_test_per_class_split(dataset, seed):
    rng = random.Random(seed)
    by_class = {}
    for index, (_path, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(index)

    train_indices = []
    test_indices = []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        test_indices.append(indices[0])
        train_indices.extend(indices[1:])

    rng.shuffle(train_indices)
    test_indices.sort(key=lambda i: dataset.samples[i][1])
    return train_indices, test_indices


def build_transforms(processor, image_size, random_erasing_p):
    mean = getattr(processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(processor, "image_std", [0.5, 0.5, 0.5])
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.65, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomErasing(p=random_erasing_p, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Resize(size=(image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def collate_fn(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(paths)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=model.training_label_smoothing)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _paths in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(pixel_values=images)
            loss = criterion(output.logits, labels)
        scaler.scale(loss).backward()
        if model.training_grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.training_grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = output.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def configure_trainable_layers(model, unfreeze_last_n_blocks):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    if unfreeze_last_n_blocks <= 0:
        return

    vit = getattr(model, "vit", None)
    encoder = getattr(vit, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        for param in model.parameters():
            param.requires_grad = True
        return

    for block in layers[-unfreeze_last_n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    layernorm = getattr(vit, "layernorm", None)
    if layernorm is not None:
        for param in layernorm.parameters():
            param.requires_grad = True


@torch.inference_mode()
def evaluate(model, loader, device, class_names):
    model.eval()
    y_true = []
    y_pred = []
    rows = []
    for images, labels, paths in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(pixel_values=images).logits
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        for path, true, pred in zip(paths, labels.cpu().tolist(), preds.cpu().tolist()):
            rows.append(
                {
                    "path": path,
                    "true_label": true,
                    "pred_label": pred,
                    "true_name": class_names[true],
                    "pred_name": class_names[pred],
                }
            )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return metrics, report, matrix, rows


def save_split(output_dir, dataset, train_indices, test_indices):
    with (output_dir / "split_train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "class_name"])
        for index in train_indices:
            path, label = dataset.samples[index]
            writer.writerow([path, label, dataset.classes[label]])
    with (output_dir / "split_test_one_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "class_name"])
        for index in test_indices:
            path, label = dataset.samples[index]
            writer.writerow([path, label, dataset.classes[label]])


def save_outputs(output_dir, args, dataset, history, metrics, report, matrix, predictions):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "history": history, "metrics": metrics}, f, ensure_ascii=False, indent=2)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    with (output_dir / "classes.txt").open("w", encoding="utf-8") as f:
        for index, name in enumerate(dataset.classes):
            f.write(f"{index}\t{name}\n")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "pred_label", "true_name", "pred_name"])
        writer.writeheader()
        writer.writerows(predictions)


def main(args):
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    image_size = args.image_size
    if image_size is None:
        size = getattr(processor, "size", {})
        image_size = size.get("height") or size.get("shortest_edge") or 224
    train_transform, eval_transform = build_transforms(processor, image_size, args.random_erasing)

    dataset = ImageFolderWithPaths(args.data)
    train_indices, test_indices = make_one_test_per_class_split(dataset, args.seed)
    save_split(output_dir, dataset, train_indices, test_indices)

    id2label = {i: name for i, name in enumerate(dataset.classes)}
    label2id = {name: i for i, name in id2label.items()}
    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=len(dataset.classes),
        id2label=id2label,
        label2id=label2id,
        local_files_only=args.local_files_only,
    )
    config.hidden_dropout_prob = args.dropout
    config.attention_probs_dropout_prob = args.attention_dropout
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.training_label_smoothing = args.label_smoothing
    model.training_grad_clip = args.grad_clip
    configure_trainable_layers(model, args.unfreeze_last_n_blocks)

    train_dataset = IndexSubset(dataset, train_indices, train_transform)
    test_dataset = IndexSubset(dataset, test_indices, eval_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Train: {len(train_indices)} images | Test: {len(test_indices)} images (1 par classe)")
    print(f"Modele: {args.model}")
    print(f"Image size: {image_size} | Device: {device}")
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Parametres entrainables: {trainable:,} / {total_params:,}")
    print(
        "Regularisation: "
        f"label_smoothing={args.label_smoothing}, dropout={args.dropout}, "
        f"attention_dropout={args.attention_dropout}, weight_decay={args.weight_decay}, "
        f"grad_clip={args.grad_clip}, random_erasing={args.random_erasing}, "
        f"unfreeze_last_n_blocks={args.unfreeze_last_n_blocks}"
    )

    best_f1 = -1.0
    best_state = None
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        metrics, _report, _matrix, _predictions = evaluate(model, test_loader, device, dataset.classes)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "test_accuracy": metrics["accuracy"],
            "test_f1_macro": metrics["f1_macro"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, report, matrix, predictions = evaluate(model, test_loader, device, dataset.classes)
    save_outputs(output_dir, args, dataset, history, metrics, report, matrix, predictions)
    model.save_pretrained(output_dir / "best_model")
    processor.save_pretrained(output_dir / "best_model")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Temps total: {(time.time() - start) / 60:.1f} min")
    print(f"Resultats sauvegardes dans: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT avec 1 image test par classe.")
    parser.add_argument("--data", default="Bases de données photos Dr Taleb")
    parser.add_argument("--model", default="bryanzhou008/vit-base-patch16-224-in21k-finetuned-inaturalist")
    parser.add_argument("--output-dir", default="results_vit_finetuned_one_test_per_class")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--random-erasing", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--unfreeze-last-n-blocks",
        type=int,
        default=2,
        help="0 entraine seulement la tete; 2-4 regularise mieux qu'un full fine-tuning sur petit dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

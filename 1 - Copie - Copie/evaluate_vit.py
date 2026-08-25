import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from timm.data import create_transform, resolve_model_data_config
from timm.utils.model import reparameterize_model
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import timm

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError:
    AutoImageProcessor = None
    AutoModelForImageClassification = None


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


class SafeImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, is_valid_file=self._is_valid_file)

    @staticmethod
    def _is_valid_file(path):
        return Path(path).suffix.lower() in IMAGE_EXTENSIONS

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception:
            sample = Image.new("RGB", (224, 224), color=(0, 0, 0))
        if sample.mode != "RGB":
            sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class DropPathDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target, _path = self.dataset[index]
        return image, target


def build_backbone(model_name, checkpoint, device):
    if checkpoint:
        model = timm.create_model(model_name, pretrained=False, checkpoint_path=checkpoint, num_classes=0)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = reparameterize_model(model)
    model.eval().to(device)
    return model


class HfFeatureBackbone(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if AutoModelForImageClassification is None:
            raise RuntimeError("Le package transformers n'est pas installe.")
        self.model = AutoModelForImageClassification.from_pretrained(model_name)

    def forward(self, images):
        output = self.model(pixel_values=images, output_hidden_states=True)
        hidden = output.hidden_states[-1]
        if hidden.ndim == 3:
            return hidden[:, 0]
        return hidden


def build_hf_backbone(model_name, device):
    model = HfFeatureBackbone(model_name)
    model.eval().to(device)
    return model


def build_hf_transform(model_name):
    if AutoImageProcessor is None:
        raise RuntimeError("Le package transformers n'est pas installe.")
    processor = AutoImageProcessor.from_pretrained(model_name)

    def transform(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    return transform


def make_loader(dataset, batch_size, workers, indices=None, shuffle=False):
    subset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def extract_features(model, loader, device):
    features = []
    targets = []
    paths = []
    for images, labels, batch_paths in tqdm(loader, desc="Extraction features"):
        images = images.to(device, non_blocking=True)
        output = model(images)
        if isinstance(output, (list, tuple)):
            output = output[0]
        features.append(output.cpu().numpy())
        targets.append(labels.numpy())
        paths.extend(batch_paths)
    return np.concatenate(features), np.concatenate(targets), paths


def split_indices(labels, test_size, seed):
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    valid_for_stratify = np.all(counts[counts > 0] >= 2)
    if valid_for_stratify:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return np.asarray(train_idx), np.asarray(test_idx)


def compute_metrics(y_true, y_pred, class_names):
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
    return metrics, report, matrix


def save_outputs(output_dir, args, class_names, metrics, report, matrix, predictions):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "metrics": metrics}, f, ensure_ascii=False, indent=2)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    with (output_dir / "classes.txt").open("w", encoding="utf-8") as f:
        for index, name in enumerate(class_names):
            f.write(f"{index}\t{name}\n")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8") as f:
        f.write("path,true_label,pred_label,true_name,pred_name\n")
        for row in predictions:
            f.write(
                f"\"{row['path']}\",{row['true_label']},{row['pred_label']},"
                f"\"{row['true_name']}\",\"{row['pred_name']}\"\n"
            )


def run_linear_probe(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.backend == "hf":
        model = build_hf_backbone(args.model, device)
        transform = build_hf_transform(args.model)
    else:
        model = build_backbone(args.model, args.checkpoint, device)
        data_config = resolve_model_data_config(model)
        transform = create_transform(**data_config, is_training=False)

    dataset = SafeImageFolder(args.data, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(f"Aucune image trouvee dans {args.data}")

    labels = [target for _path, target in dataset.samples]
    train_idx, test_idx = split_indices(labels, args.test_size, args.seed)

    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Train: {len(train_idx)} images | Test: {len(test_idx)} images")
    print(f"Backbone: {args.model}")
    print(f"Device: {device}")

    train_loader = make_loader(dataset, args.batch_size, args.workers, train_idx)
    test_loader = make_loader(dataset, args.batch_size, args.workers, test_idx)

    x_train, y_train, _train_paths = extract_features(model, train_loader, device)
    x_test, y_test, test_paths = extract_features(model, test_loader, device)

    classifier = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        class_weight="balanced" if args.class_weight_balanced else None,
        n_jobs=-1,
        random_state=args.seed,
    )
    start = time.time()
    classifier.fit(x_train, y_train)
    print(f"Linear probe entraine en {time.time() - start:.1f}s")

    y_pred = classifier.predict(x_test)
    metrics, report, matrix = compute_metrics(y_test, y_pred, dataset.classes)
    predictions = [
        {
            "path": path,
            "true_label": int(true),
            "pred_label": int(pred),
            "true_name": dataset.classes[int(true)],
            "pred_name": dataset.classes[int(pred)],
        }
        for path, true, pred in zip(test_paths, y_test, y_pred)
    ]

    save_outputs(Path(args.output_dir), args, dataset.classes, metrics, report, matrix, predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Resultats sauvegardes dans: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluer un ViT pre-entraine comme backbone sur un dataset ImageFolder."
    )
    parser.add_argument("--data", default="Bases de données photos Dr Taleb", help="Dossier racine du dataset.")
    parser.add_argument(
        "--backend",
        choices=["timm", "hf"],
        default="timm",
        help="Backend du modele: timm ou hf pour Hugging Face Transformers.",
    )
    parser.add_argument(
        "--model",
        default="vit_base_patch16_224.augreg2_in21k_ft_in1k",
        help="Nom timm du ViT. Utiliser ici le nom de votre ViT pre-entraine iNaturalist si disponible.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint local .pth/.pt optionnel.")
    parser.add_argument("--output-dir", default="results_vit_eval", help="Dossier de sortie.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, cpu, ou vide pour auto.")
    parser.add_argument("--c", type=float, default=1.0, help="Regularisation inverse LogisticRegression.")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--class-weight-balanced", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_linear_probe(parse_args())

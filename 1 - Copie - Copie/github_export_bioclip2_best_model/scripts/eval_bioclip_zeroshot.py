import argparse
import csv
import json
import random
import re
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
from tqdm import tqdm

import open_clip


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}

PROMPT_TEMPLATES = [
    "a photo of {name}",
    "a photo of the plant {name}",
    "a close-up photo of {name}",
    "a field photo of {name}",
    "a botanical photo of {name}",
]

BINOMIAL_PLANT_TEMPLATES = [
    "a photo of {binomial}",
    "a photo of the plant {binomial}",
    "a field photo of the weed species {binomial}",
    "a botanical photo of {binomial}",
    "a close-up photo of the leaves or flowers of {binomial}",
]


def binomial_name(text):
    cleaned = re.sub(r"\([^)]*\)", " ", text)
    parts = re.findall(r"[A-Z]?[a-zA-Z]+(?:-[a-zA-Z]+)?", cleaned)
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return re.sub(r"\s+", " ", text).strip()


class SafeImageFolder(ImageFolder):
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


class IndexDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


def one_image_per_class_indices(dataset, seed):
    rng = random.Random(seed)
    by_class = {}
    for index, (_path, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(index)
    selected = []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        selected.append(indices[0])
    return selected


@torch.inference_mode()
def encode_text(model, tokenizer, class_names, device):
    templates = BINOMIAL_PLANT_TEMPLATES if encode_text.prompt_mode == "binomial_plant" else PROMPT_TEMPLATES
    all_features = []
    for template in templates:
        prompts = [template.format(name=name, binomial=binomial_name(name)) for name in class_names]
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features)
    features = torch.stack(all_features).mean(dim=0)
    return features / features.norm(dim=-1, keepdim=True)


encode_text.prompt_mode = "full_name_plant"


@torch.inference_mode()
def evaluate(model, loader, text_features, device, class_names):
    model.eval()
    y_true = []
    y_pred = []
    top5_correct = 0
    rows = []
    for images, labels, paths in tqdm(loader, desc="BioCLIP zero-shot"):
        images = images.to(device, non_blocking=True)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        top5 = logits.topk(k=min(5, len(class_names)), dim=1).indices
        preds = top5[:, 0]

        labels_cpu = labels.cpu()
        y_true.extend(labels_cpu.tolist())
        y_pred.extend(preds.cpu().tolist())
        top5_correct += (top5.cpu() == labels_cpu.unsqueeze(1)).any(dim=1).sum().item()
        probs = logits.softmax(dim=1)
        for path, true, pred, prob in zip(paths, labels_cpu.tolist(), preds.cpu().tolist(), probs.cpu()):
            rows.append(
                {
                    "path": path,
                    "true_label": true,
                    "pred_label": pred,
                    "true_name": class_names[true],
                    "pred_name": class_names[pred],
                    "pred_score": float(prob[pred]),
                }
            )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy_top1": float(accuracy_score(y_true, y_pred)),
        "accuracy_top5": float(top5_correct / max(len(y_true), 1)),
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


def save_outputs(output_dir, args, class_names, metrics, report, matrix, predictions, indices, dataset):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "metrics": metrics}, f, ensure_ascii=False, indent=2)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    with (output_dir / "classes.txt").open("w", encoding="utf-8") as f:
        for index, name in enumerate(class_names):
            f.write(f"{index}\t{name}\n")
    with (output_dir / "split_eval.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "class_name"])
        for index in indices:
            path, label = dataset.samples[index]
            writer.writerow([path, label, class_names[label]])
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "true_label", "pred_label", "true_name", "pred_name", "pred_score"],
        )
        writer.writeheader()
        writer.writerows(predictions)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _train_preprocess, preprocess = open_clip.create_model_and_transforms(
        args.arch,
        pretrained=args.pretrained,
    )
    tokenizer = open_clip.get_tokenizer(args.arch)
    model.eval().to(device)
    encode_text.prompt_mode = args.prompt_mode

    dataset = SafeImageFolder(args.data, transform=preprocess)
    indices = one_image_per_class_indices(dataset, args.seed) if args.sample == "one_per_class" else list(range(len(dataset)))
    loader = DataLoader(
        IndexDataset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    text_features = encode_text(model, tokenizer, dataset.classes, device)
    metrics, report, matrix, predictions = evaluate(model, loader, text_features, device, dataset.classes)
    metrics["num_classes"] = len(dataset.classes)
    metrics["num_eval_images"] = len(indices)
    metrics["prompt_templates"] = (
        BINOMIAL_PLANT_TEMPLATES if args.prompt_mode == "binomial_plant" else PROMPT_TEMPLATES
    )
    save_outputs(Path(args.output_dir), args, dataset.classes, metrics, report, matrix, predictions, indices, dataset)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Resultats sauvegardes dans: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation zero-shot BioCLIP sur les classes du dataset.")
    parser.add_argument("--data", default="Bases de données photos Dr Taleb")
    parser.add_argument("--arch", default="hf-hub:imageomics/bioclip-2")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--output-dir", default="results_bioclip2_zeroshot_one_test_per_class")
    parser.add_argument("--sample", choices=["one_per_class", "all"], default="one_per_class")
    parser.add_argument("--prompt-mode", choices=["full_name_plant", "binomial_plant"], default="full_name_plant")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

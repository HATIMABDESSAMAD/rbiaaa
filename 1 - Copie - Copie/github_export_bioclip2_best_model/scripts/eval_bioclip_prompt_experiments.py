import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import open_clip


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def binomial_name(text):
    cleaned = re.sub(r"\([^)]*\)", " ", text)
    parts = re.findall(r"[A-Z]?[a-zA-Z]+(?:-[a-zA-Z]+)?", cleaned)
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return normalize_spaces(text)


def remove_authors(text):
    return binomial_name(text)


PROMPT_SETS = {
    "full_name_basic": [
        "a photo of {name}",
    ],
    "full_name_plant_templates": [
        "a photo of {name}",
        "a photo of the plant {name}",
        "a close-up photo of {name}",
        "a field photo of {name}",
        "a botanical photo of {name}",
    ],
    "binomial_basic": [
        "a photo of {binomial}",
    ],
    "binomial_plant_templates": [
        "a photo of {binomial}",
        "a photo of the plant {binomial}",
        "a field photo of the weed species {binomial}",
        "a botanical photo of {binomial}",
        "a close-up photo of the leaves or flowers of {binomial}",
    ],
    "full_plus_binomial_ensemble": [
        "a photo of {name}",
        "a photo of the plant {name}",
        "a botanical photo of {name}",
        "a photo of {binomial}",
        "a photo of the plant {binomial}",
        "a field photo of the weed species {binomial}",
        "a botanical photo of {binomial}",
    ],
}


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


@torch.inference_mode()
def encode_images(model, loader, device):
    features = []
    labels = []
    paths = []
    for images, targets, batch_paths in tqdm(loader, desc="Encode images"):
        images = images.to(device, non_blocking=True)
        batch_features = model.encode_image(images)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features.cpu())
        labels.extend(targets.tolist())
        paths.extend(batch_paths)
    return torch.cat(features), np.asarray(labels), paths


@torch.inference_mode()
def encode_prompt_set(model, tokenizer, class_names, prompt_templates, device):
    template_features = []
    for template in prompt_templates:
        prompts = [
            template.format(name=name, binomial=binomial_name(name), clean=remove_authors(name))
            for name in class_names
        ]
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        template_features.append(features)
    text_features = torch.stack(template_features).mean(dim=0)
    return text_features / text_features.norm(dim=-1, keepdim=True)


def compute_metrics(y_true, y_pred, top5):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy_top1": float(accuracy_score(y_true, y_pred)),
        "accuracy_top5": float(np.mean([true in row for true, row in zip(y_true, top5)])),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model, _train_preprocess, preprocess = open_clip.create_model_and_transforms(args.arch, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.arch)
    model.eval().to(device)

    dataset = SafeImageFolder(args.data, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    image_features, y_true, paths = encode_images(model, loader, device)
    image_features_device = image_features.to(device)

    rows = []
    all_metrics = {}
    best_name = None
    best_acc = -1.0
    best_predictions = None

    for name, templates in PROMPT_SETS.items():
        text_features = encode_prompt_set(model, tokenizer, dataset.classes, templates, device)
        logits = 100.0 * image_features_device @ text_features.T
        topk = logits.topk(k=min(5, len(dataset.classes)), dim=1).indices.cpu().numpy()
        y_pred = topk[:, 0]
        metrics = compute_metrics(y_true, y_pred, topk)
        metrics["num_classes"] = len(dataset.classes)
        metrics["num_eval_images"] = len(y_true)
        metrics["templates"] = templates
        all_metrics[name] = metrics
        rows.append({"prompt_set": name, **{k: v for k, v in metrics.items() if not isinstance(v, list)}})
        if metrics["accuracy_top1"] > best_acc:
            best_acc = metrics["accuracy_top1"]
            best_name = name
            probs = logits.softmax(dim=1).detach().cpu().numpy()
            best_predictions = [
                {
                    "path": path,
                    "true_label": int(true),
                    "pred_label": int(pred),
                    "true_name": dataset.classes[int(true)],
                    "pred_name": dataset.classes[int(pred)],
                    "pred_score": float(probs[i, int(pred)]),
                }
                for i, (path, true, pred) in enumerate(zip(paths, y_true, y_pred))
            ]
        print(name, json.dumps(metrics, ensure_ascii=False))

    with (output_dir / "prompt_experiment_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_prompt_set": best_name, "results": all_metrics}, f, ensure_ascii=False, indent=2)
    with (output_dir / "prompt_experiment_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "best_prompt_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "true_label", "pred_label", "true_name", "pred_name", "pred_score"],
        )
        writer.writeheader()
        writer.writerows(best_predictions)
    print(f"Best prompt set: {best_name}")
    print(f"Resultats sauvegardes dans: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Comparer plusieurs strategies de prompts BioCLIP.")
    parser.add_argument("--data", default="Bases de données photos Dr Taleb")
    parser.add_argument("--arch", default="hf-hub:imageomics/bioclip-2")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--output-dir", default="results_bioclip2_prompt_experiments_all_dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

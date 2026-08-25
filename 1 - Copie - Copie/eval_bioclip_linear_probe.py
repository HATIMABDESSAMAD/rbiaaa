import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import open_clip


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
    for images, targets, batch_paths in tqdm(loader, desc="Encode BioCLIP features"):
        images = images.to(device, non_blocking=True)
        batch_features = model.encode_image(images)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features.cpu().numpy())
        labels.extend(targets.tolist())
        paths.extend(batch_paths)
    return np.concatenate(features), np.asarray(labels), paths


def one_test_per_class_split(y, seed):
    rng = random.Random(seed)
    train = []
    test = []
    for label in sorted(set(y.tolist())):
        indices = np.where(y == label)[0].tolist()
        rng.shuffle(indices)
        test.append(indices[0])
        train.extend(indices[1:])
    return np.asarray(train), np.asarray(test)


def compute_metrics(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def nearest_centroid_predict(x_train, y_train, x_test):
    centroids = []
    labels = []
    for label in sorted(set(y_train.tolist())):
        center = x_train[y_train == label].mean(axis=0)
        center = center / max(np.linalg.norm(center), 1e-12)
        centroids.append(center)
        labels.append(label)
    centroids = np.stack(centroids)
    labels = np.asarray(labels)
    x_test_norm = x_test / np.maximum(np.linalg.norm(x_test, axis=1, keepdims=True), 1e-12)
    scores = x_test_norm @ centroids.T
    return labels[scores.argmax(axis=1)]


def summarize(rows, metric_names):
    summary = {}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model, _train_preprocess, preprocess = open_clip.create_model_and_transforms(args.arch, pretrained=args.pretrained)
    model.eval().to(device)
    dataset = SafeImageFolder(args.data, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    x, y, paths = encode_images(model, loader, device)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    all_rows = []
    predictions_rows = []

    seeds = [args.seed + i for i in range(args.repeats)]
    c_values = [float(v) for v in args.c_values.split(",")]
    for c_value in c_values:
        for seed in seeds:
            train_idx, test_idx = one_test_per_class_split(y, seed)
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed,
                ),
            )
            classifier.fit(x[train_idx], y[train_idx])
            pred = classifier.predict(x[test_idx])
            metrics = compute_metrics(y[test_idx], pred)
            row = {
                "method": "logistic_regression",
                "c": c_value,
                "seed": seed,
                "train_images": int(len(train_idx)),
                "test_images": int(len(test_idx)),
                **metrics,
            }
            all_rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
            for index, pred_label in zip(test_idx, pred):
                predictions_rows.append(
                    {
                        "c": c_value,
                        "seed": seed,
                        "path": paths[index],
                        "true_label": int(y[index]),
                        "pred_label": int(pred_label),
                        "true_name": dataset.classes[int(y[index])],
                        "pred_name": dataset.classes[int(pred_label)],
                    }
                )

    for seed in seeds:
        train_idx, test_idx = one_test_per_class_split(y, seed)
        pred = nearest_centroid_predict(x[train_idx], y[train_idx], x[test_idx])
        metrics = compute_metrics(y[test_idx], pred)
        row = {
            "method": "nearest_centroid",
            "c": 0.0,
            "seed": seed,
            "train_images": int(len(train_idx)),
            "test_images": int(len(test_idx)),
            **metrics,
        }
        all_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        for index, pred_label in zip(test_idx, pred):
            predictions_rows.append(
                {
                    "c": 0.0,
                    "seed": seed,
                    "path": paths[index],
                    "true_label": int(y[index]),
                    "pred_label": int(pred_label),
                    "true_name": dataset.classes[int(y[index])],
                    "pred_name": dataset.classes[int(pred_label)],
                }
            )

    summaries = []
    for c_value in c_values:
        rows = [row for row in all_rows if row["c"] == c_value]
        summaries.append({"method": "logistic_regression", "c": c_value, **summarize(rows, metric_names)})
    rows = [row for row in all_rows if row["method"] == "nearest_centroid"]
    summaries.append({"method": "nearest_centroid", "c": 0.0, **summarize(rows, metric_names)})
    best = max(summaries, key=lambda row: row["accuracy_mean"])

    with (output_dir / "linear_probe_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "linear_probe_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "linear_probe_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["c", "seed", "path", "true_label", "pred_label", "true_name", "pred_name"],
        )
        writer.writeheader()
        writer.writerows(predictions_rows)
    with (output_dir / "linear_probe_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "num_classes": len(dataset.classes),
                "num_images": len(dataset),
                "best_by_accuracy_mean": best,
                "summaries": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Best summary:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Resultats sauvegardes dans: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe sur features BioCLIP.")
    parser.add_argument("--data", default="Bases de données photos Dr Taleb")
    parser.add_argument("--arch", default="hf-hub:imageomics/bioclip-2")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--output-dir", default="results_bioclip2_linear_probe_repeated_one_test_per_class")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--c-values", default="0.01,0.1,1,10")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

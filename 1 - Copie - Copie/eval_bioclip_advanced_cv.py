import argparse
import csv
import json
import random
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


def binomial_name(text):
    cleaned = re.sub(r"\([^)]*\)", " ", text)
    parts = re.findall(r"[A-Z]?[a-zA-Z]+(?:-[a-zA-Z]+)?", cleaned)
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return re.sub(r"\s+", " ", text).strip()


PROMPT_CANDIDATES = [
    ("binomial_photo", "a photo of {binomial}"),
    ("binomial_plant", "a photo of the plant {binomial}"),
    ("binomial_weed", "a field photo of the weed species {binomial}"),
    ("binomial_botanical", "a botanical photo of {binomial}"),
    ("binomial_closeup", "a close-up photo of {binomial}"),
    ("binomial_leaf_flower", "a close-up photo of the leaves or flowers of {binomial}"),
    ("binomial_seedling", "a photo of the seedling of {binomial}"),
    ("binomial_flower", "a photo of the flower of {binomial}"),
    ("binomial_leaf", "a photo of the leaf of {binomial}"),
    ("binomial_inflorescence", "a photo of the inflorescence of {binomial}"),
    ("full_photo", "a photo of {name}"),
    ("full_plant", "a photo of the plant {name}"),
    ("full_botanical", "a botanical photo of {name}"),
]

BASELINE_TEMPLATES = [
    "a photo of {binomial}",
    "a photo of the plant {binomial}",
    "a field photo of the weed species {binomial}",
    "a botanical photo of {binomial}",
    "a close-up photo of the leaves or flowers of {binomial}",
]


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
        features.append(batch_features.cpu().numpy())
        labels.extend(targets.tolist())
        paths.extend(batch_paths)
    return np.concatenate(features), np.asarray(labels), paths


@torch.inference_mode()
def encode_candidate_texts(model, tokenizer, class_names, device):
    candidate_features = []
    candidate_names = []
    for candidate_name, template in PROMPT_CANDIDATES:
        prompts = [
            template.format(name=name, binomial=binomial_name(name))
            for name in class_names
        ]
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        candidate_features.append(features.cpu().numpy())
        candidate_names.append(candidate_name)
    return np.stack(candidate_features), candidate_names


@torch.inference_mode()
def encode_baseline_text(model, tokenizer, class_names, device):
    all_features = []
    for template in BASELINE_TEMPLATES:
        prompts = [
            template.format(name=name, binomial=binomial_name(name))
            for name in class_names
        ]
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features)
    features = torch.stack(all_features).mean(dim=0)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


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


def predict_text(x, text_features):
    scores = x @ text_features.T
    return scores.argmax(axis=1), scores


def select_prompt_per_class(x_train, y_train, candidate_texts):
    num_candidates, num_classes, _dim = candidate_texts.shape
    selected = []
    for cls in range(num_classes):
        cls_indices = np.where(y_train == cls)[0]
        best_candidate = 0
        best_score = -1.0
        for cand in range(num_candidates):
            text_features = candidate_texts[cand]
            pred, _scores = predict_text(x_train[cls_indices], text_features)
            score = float(np.mean(pred == cls))
            if score > best_score:
                best_score = score
                best_candidate = cand
        selected.append(best_candidate)
    selected_text = np.stack([candidate_texts[cand, cls] for cls, cand in enumerate(selected)])
    selected_text = selected_text / np.maximum(np.linalg.norm(selected_text, axis=1, keepdims=True), 1e-12)
    return np.asarray(selected), selected_text


def compute_centroids(x_train, y_train, num_classes):
    centroids = []
    for cls in range(num_classes):
        center = x_train[y_train == cls].mean(axis=0)
        center = center / max(np.linalg.norm(center), 1e-12)
        centroids.append(center)
    return np.stack(centroids)


def loo_centroid_scores(x_train, y_train, num_classes):
    scores = np.zeros((len(y_train), num_classes), dtype=np.float32)
    for i in range(len(y_train)):
        centroids = []
        for cls in range(num_classes):
            cls_indices = np.where(y_train == cls)[0]
            cls_indices = cls_indices[cls_indices != i]
            if len(cls_indices) == 0:
                center = x_train[y_train == cls].mean(axis=0)
            else:
                center = x_train[cls_indices].mean(axis=0)
            center = center / max(np.linalg.norm(center), 1e-12)
            centroids.append(center)
        scores[i] = x_train[i] @ np.stack(centroids).T
    return scores


def softmax(scores, temperature=1.0):
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def choose_alpha_on_train(text_scores_train, centroid_scores_train, y_train):
    best_alpha = 0.0
    best_acc = -1.0
    text_probs = softmax(text_scores_train)
    centroid_probs = softmax(centroid_scores_train)
    for alpha in np.linspace(0.0, 1.0, 21):
        combined = alpha * text_probs + (1.0 - alpha) * centroid_probs
        acc = accuracy_score(y_train, combined.argmax(axis=1))
        if acc > best_acc:
            best_acc = acc
            best_alpha = float(alpha)
    return best_alpha, float(best_acc)


def metrics(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }


def summarize(rows, method):
    selected = [row for row in rows if row["method"] == method]
    out = {"method": method}
    for key in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        values = np.asarray([row[key] for row in selected], dtype=float)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    return out


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _train_preprocess, preprocess = open_clip.create_model_and_transforms(args.arch, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.arch)
    model.eval().to(device)

    dataset = SafeImageFolder(args.data, transform=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    x, y, paths = encode_images(model, loader, device)
    num_classes = len(dataset.classes)
    baseline_text = encode_baseline_text(model, tokenizer, dataset.classes, device)
    candidate_texts, candidate_names = encode_candidate_texts(model, tokenizer, dataset.classes, device)

    rows = []
    prompt_selection_rows = []
    prediction_rows = []
    for seed in [args.seed + i for i in range(args.repeats)]:
        train_idx, test_idx = one_test_per_class_split(y, seed)
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        baseline_pred, baseline_test_scores = predict_text(x_test, baseline_text)
        rows.append({"method": "baseline_binomial_prompts", "seed": seed, **metrics(y_test, baseline_pred)})

        selected_candidate_ids, selected_text = select_prompt_per_class(x_train, y_train, candidate_texts)
        selected_pred, selected_test_scores = predict_text(x_test, selected_text)
        rows.append({"method": "class_adaptive_prompts", "seed": seed, **metrics(y_test, selected_pred)})
        for cls, cand_id in enumerate(selected_candidate_ids):
            prompt_selection_rows.append(
                {
                    "seed": seed,
                    "class_id": cls,
                    "class_name": dataset.classes[cls],
                    "selected_prompt": candidate_names[int(cand_id)],
                    "selected_template": PROMPT_CANDIDATES[int(cand_id)][1],
                }
            )

        centroids = compute_centroids(x_train, y_train, num_classes)
        centroid_test_scores = x_test @ centroids.T
        loo_scores = loo_centroid_scores(x_train, y_train, num_classes)
        selected_train_scores = x_train @ selected_text.T
        alpha, train_alpha_acc = choose_alpha_on_train(selected_train_scores, loo_scores, y_train)
        combined_probs = alpha * softmax(selected_test_scores) + (1.0 - alpha) * softmax(centroid_test_scores)
        ensemble_pred = combined_probs.argmax(axis=1)
        rows.append(
            {
                "method": "adaptive_prompt_plus_centroid",
                "seed": seed,
                "alpha_text": alpha,
                "inner_train_accuracy": train_alpha_acc,
                **metrics(y_test, ensemble_pred),
            }
        )

        for idx, pred in zip(test_idx, ensemble_pred):
            prediction_rows.append(
                {
                    "seed": seed,
                    "path": paths[idx],
                    "true_label": int(y[idx]),
                    "pred_label": int(pred),
                    "true_name": dataset.classes[int(y[idx])],
                    "pred_name": dataset.classes[int(pred)],
                }
            )
        print(json.dumps(rows[-3:], ensure_ascii=False))

    summaries = [summarize(rows, method) for method in ["baseline_binomial_prompts", "class_adaptive_prompts", "adaptive_prompt_plus_centroid"]]
    best = max(summaries, key=lambda row: row["accuracy_mean"])

    with (output_dir / "advanced_cv_runs.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "advanced_cv_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "selected_prompts_by_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prompt_selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prompt_selection_rows)
    with (output_dir / "ensemble_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (output_dir / "advanced_cv_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "num_classes": num_classes, "num_images": len(dataset), "best": best, "summaries": summaries}, f, ensure_ascii=False, indent=2)
    print("Best:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Resultats sauvegardes dans: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ameliorations BioCLIP rigoureuses avec validation croisee.")
    parser.add_argument("--data", default="Bases de données photos Dr Taleb")
    parser.add_argument("--arch", default="hf-hub:imageomics/bioclip-2")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--output-dir", default="results_bioclip2_advanced_cv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

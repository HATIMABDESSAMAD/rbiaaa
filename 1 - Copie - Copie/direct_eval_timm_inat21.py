import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import timm


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def normalize_text(text):
    text = (
        text.replace("ï", "i")
        .replace("ö", "o")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ç", "c")
    )
    return re.sub(r"\s+", " ", text.strip()).lower()


def binomial_name(text):
    text = re.sub(r"\([^)]*\)", " ", normalize_text(text))
    text = text.replace("&", " ")
    parts = re.findall(r"[a-z]+(?:-[a-z]+)?", text)
    if len(parts) >= 2:
        return " ".join(parts[:2])
    return normalize_text(text)


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
    def __init__(self, dataset, indices, mapped_label_by_dataset_label):
        self.dataset = dataset
        self.indices = list(indices)
        self.mapped_label_by_dataset_label = mapped_label_by_dataset_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, dataset_label, path = self.dataset[self.indices[item]]
        global_label = self.mapped_label_by_dataset_label[dataset_label]
        return image, dataset_label, global_label, path


def one_image_per_class_indices(dataset, matched_dataset_labels, seed):
    rng = random.Random(seed)
    by_label = {label: [] for label in matched_dataset_labels}
    for index, (_path, label) in enumerate(dataset.samples):
        if label in by_label:
            by_label[label].append(index)
    selected = []
    for label in sorted(by_label):
        indices = by_label[label]
        rng.shuffle(indices)
        selected.append(indices[0])
    return selected


def all_matched_indices(dataset, matched_dataset_labels):
    matched = set(matched_dataset_labels)
    return [index for index, (_path, label) in enumerate(dataset.samples) if label in matched]


def build_mapping(dataset, label_names):
    i_nat_binomial_to_global = {}
    for global_id, label_name in enumerate(label_names):
        i_nat_binomial_to_global.setdefault(binomial_name(label_name), global_id)

    mapped = {}
    missing = []
    for dataset_id, class_name in enumerate(dataset.classes):
        global_id = i_nat_binomial_to_global.get(binomial_name(class_name))
        if global_id is None:
            missing.append(class_name)
        else:
            mapped[dataset_id] = global_id
    return mapped, missing


@torch.inference_mode()
def evaluate(model, loader, device, restricted_global_ids):
    model.eval()
    y_dataset_true = []
    y_global_true = []
    y_global_pred_full = []
    y_global_pred_restricted = []
    paths = []
    restricted_tensor = torch.tensor(restricted_global_ids, device=device)

    for images, dataset_labels, global_labels, batch_paths in tqdm(loader, desc="Eval direct iNat21"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        pred_full = logits.argmax(dim=1)
        pred_restricted = restricted_tensor[logits.index_select(1, restricted_tensor).argmax(dim=1)]

        y_dataset_true.extend(dataset_labels.tolist())
        y_global_true.extend(global_labels.tolist())
        y_global_pred_full.extend(pred_full.cpu().tolist())
        y_global_pred_restricted.extend(pred_restricted.cpu().tolist())
        paths.extend(batch_paths)

    return y_dataset_true, y_global_true, y_global_pred_full, y_global_pred_restricted, paths


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = timm.create_model(args.model, pretrained=True)
    model.eval().to(device)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    dataset = SafeImageFolder(args.data, transform=transform)

    label_names = model.pretrained_cfg.get("label_names")
    if not label_names:
        raise RuntimeError("Ce modele ne fournit pas label_names dans pretrained_cfg.")

    mapped, missing = build_mapping(dataset, label_names)
    if args.sample == "one_per_class":
        indices = one_image_per_class_indices(dataset, mapped.keys(), args.seed)
    else:
        indices = all_matched_indices(dataset, mapped.keys())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        IndexDataset(dataset, indices, mapped),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    restricted_global_ids = sorted(set(mapped.values()))
    y_dataset_true, y_global_true, pred_full, pred_restricted, paths = evaluate(
        model, loader, device, restricted_global_ids
    )

    metrics = {
        "num_dataset_classes": len(dataset.classes),
        "num_matched_classes": len(mapped),
        "num_missing_classes": len(missing),
        "num_eval_images": len(y_global_true),
        "accuracy_full_10000_labels": float(accuracy_score(y_global_true, pred_full)),
        "accuracy_restricted_to_matched_classes": float(accuracy_score(y_global_true, pred_restricted)),
    }

    matched_dataset_labels = sorted(mapped)
    global_to_dataset = {global_id: dataset_id for dataset_id, global_id in mapped.items()}
    y_pred_dataset_restricted = [global_to_dataset.get(pred, -1) for pred in pred_restricted]
    report = classification_report(
        y_dataset_true,
        y_pred_dataset_restricted,
        labels=matched_dataset_labels,
        target_names=[dataset.classes[i] for i in matched_dataset_labels],
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_dataset_true, y_pred_dataset_restricted, labels=matched_dataset_labels)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "metrics": metrics}, f, ensure_ascii=False, indent=2)
    with (output_dir / "missing_classes.txt").open("w", encoding="utf-8") as f:
        for name in missing:
            f.write(name + "\n")
    with (output_dir / "matched_classes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_label", "dataset_class", "inat21_global_label", "inat21_class"])
        for dataset_id in matched_dataset_labels:
            global_id = mapped[dataset_id]
            writer.writerow([dataset_id, dataset.classes[dataset_id], global_id, label_names[global_id]])
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path",
                "true_dataset_class",
                "true_inat21_class",
                "pred_full_10000",
                "pred_restricted",
                "correct_full",
                "correct_restricted",
            ]
        )
        for path, true_dataset, true_global, full_pred, restricted_pred in zip(
            paths, y_dataset_true, y_global_true, pred_full, pred_restricted
        ):
            writer.writerow(
                [
                    path,
                    dataset.classes[true_dataset],
                    label_names[true_global],
                    label_names[full_pred],
                    label_names[restricted_pred],
                    int(full_pred == true_global),
                    int(restricted_pred == true_global),
                ]
            )
    with (output_dir / "classification_report_restricted.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savetxt(output_dir / "confusion_matrix_restricted.csv", matrix, delimiter=",", fmt="%d")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Resultats sauvegardes dans: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation directe du ViT iNat21 sans fine-tuning.")
    parser.add_argument("--data", default="Bases de données photos Dr Taleb")
    parser.add_argument("--model", default="hf_hub:timm/vit_large_patch14_clip_336.laion2b_ft_augreg_inat21")
    parser.add_argument("--output-dir", default="results_vit_inat21_direct_one_test_per_matched_class")
    parser.add_argument("--sample", choices=["one_per_class", "all"], default="one_per_class")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

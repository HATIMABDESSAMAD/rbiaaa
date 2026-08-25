import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}


class ImagePathDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = row["image_path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, idx


class PilImagePathDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB").copy()
        return image, idx


def pil_collate(batch):
    images, indexes = zip(*batch)
    return list(images), torch.tensor(indexes, dtype=torch.long)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan_dataset(dataset_dir, output_csv, keep_cross_class_duplicates=False):
    dataset_dir = Path(dataset_dir)
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    class_to_id = {p.name: i for i, p in enumerate(class_dirs)}

    rows = []
    for class_dir in class_dirs:
        for path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    image.verify()
                rows.append(
                    {
                        "image_path": str(path),
                        "class_name": class_dir.name,
                        "class_id": class_to_id[class_dir.name],
                        "file_name": path.name,
                        "extension": path.suffix,
                        "width": width,
                        "height": height,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "image_path": str(path),
                        "class_name": class_dir.name,
                        "class_id": class_to_id[class_dir.name],
                        "file_name": path.name,
                        "extension": path.suffix,
                        "width": "",
                        "height": "",
                        "size_bytes": path.stat().st_size,
                        "sha256": "",
                        "error": str(exc),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No image found in {dataset_dir}")

    df["duplicate_group_size"] = df.groupby("sha256")["sha256"].transform("count")
    df["duplicate_group_id"] = df["sha256"].where(df["duplicate_group_size"] > 1, "")
    df["duplicate_rank"] = df.groupby("sha256").cumcount()
    df["duplicate_classes"] = df.groupby("sha256")["class_name"].transform(lambda s: " | ".join(sorted(set(s))))
    df["cross_class_duplicate"] = df.groupby("sha256")["class_name"].transform("nunique") > 1
    df["same_class_duplicate_extra"] = (df["duplicate_group_size"] > 1) & (df["duplicate_rank"] > 0) & (~df["cross_class_duplicate"])
    if "error" not in df.columns:
        df["error"] = ""
    cross_ok = True if keep_cross_class_duplicates else ~df["cross_class_duplicate"]
    df["usable_for_training"] = (~df["same_class_duplicate_extra"]) & cross_ok & df["error"].eq("")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def build_feature_extractor(name, device):
    name = name.lower()
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        transform = weights.transforms()
        embedding_dim = 2048
    elif name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        model = models.vit_b_16(weights=weights)
        model.heads = torch.nn.Identity()
        transform = weights.transforms()
        embedding_dim = 768
    else:
        raise ValueError("Supported models: resnet50, vit_b_16")

    model.eval().to(device)
    return model, transform, embedding_dim


def dinov2_model_id(name):
    if name == "dinov2_small":
        return "facebook/dinov2-small"
    if name == "dinov2_base":
        return "facebook/dinov2-base"
    raise ValueError("Supported DINOv2 models: dinov2_small, dinov2_base")


@torch.inference_mode()
def extract_embeddings(df, model_name, batch_size, output_npz, tta_crops=1):
    if model_name.startswith("dinov2_"):
        return extract_dinov2_embeddings(df, model_name, batch_size, output_npz, tta_crops=tta_crops)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform, embedding_dim = build_feature_extractor(model_name, device)
    usable = df[df["usable_for_training"]].reset_index(drop=True)
    dataset = ImagePathDataset(usable.to_dict("records"), transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings = np.zeros((len(dataset), embedding_dim), dtype=np.float32)
    for images, indexes in loader:
        images = images.to(device)
        vectors = model(images).detach().cpu().numpy()
        embeddings[indexes.numpy()] = vectors

    labels = usable["class_name"].to_numpy()
    paths = usable["image_path"].to_numpy()
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, embeddings=embeddings, labels=labels, paths=paths, model_name=model_name)
    return embeddings, labels, paths


@torch.inference_mode()
def make_tta_views(image, tta_crops):
    if tta_crops <= 1:
        return [image]

    views = [image]
    width, height = image.size
    side = min(width, height)
    if side > 0:
        boxes = [
            ((width - side) // 2, (height - side) // 2, (width + side) // 2, (height + side) // 2),
            (0, 0, side, side),
            (width - side, 0, width, side),
            (0, height - side, side, height),
            (width - side, height - side, width, height),
        ]
        for box in boxes:
            views.append(image.crop(box))
            if len(views) >= tta_crops:
                return views

    views.append(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    return views[:tta_crops]


def extract_dinov2_embeddings(df, model_name, batch_size, output_npz, tta_crops=1):
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = dinov2_model_id(model_name)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)

    usable = df[df["usable_for_training"]].reset_index(drop=True)
    dataset = PilImagePathDataset(usable.to_dict("records"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=pil_collate)

    vectors = []
    ordered_indexes = []
    for images, indexes in loader:
        batch_vectors = []
        for image in images:
            views = make_tta_views(image, tta_crops)
            inputs = processor(images=views, return_tensors="pt").to(device)
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                view_vectors = outputs.pooler_output
            else:
                view_vectors = outputs.last_hidden_state[:, 0]
            view_vectors = torch.nn.functional.normalize(view_vectors, dim=1)
            image_vector = torch.nn.functional.normalize(view_vectors.mean(dim=0, keepdim=True), dim=1)
            batch_vectors.append(image_vector.cpu().numpy().astype(np.float32))
        vectors.append(np.concatenate(batch_vectors, axis=0))
        ordered_indexes.extend(indexes.numpy().tolist())

    embeddings = np.concatenate(vectors, axis=0)
    order = np.argsort(np.asarray(ordered_indexes))
    embeddings = embeddings[order]
    labels = usable["class_name"].to_numpy()
    paths = usable["image_path"].to_numpy()

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, embeddings=embeddings, labels=labels, paths=paths, model_name=model_name)
    return embeddings, labels, paths


def make_classifier(kind, svm_c=1.0):
    kind = kind.lower()
    if kind == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=svm_c, class_weight="balanced", probability=False, decision_function_shape="ovr"),
        )
    if kind == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=False, decision_function_shape="ovr"),
        )
    if kind == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1, metric="cosine"))
    raise ValueError("Supported classifiers: svm, svm_rbf, knn, proto, svm_proto")


def prototype_scores(x_train, y_train, x_test, num_classes):
    x_train = x_train / np.maximum(np.linalg.norm(x_train, axis=1, keepdims=True), 1e-12)
    x_test = x_test / np.maximum(np.linalg.norm(x_test, axis=1, keepdims=True), 1e-12)
    prototypes = np.zeros((num_classes, x_train.shape[1]), dtype=np.float32)
    valid = np.zeros(num_classes, dtype=bool)
    for class_id in range(num_classes):
        class_vectors = x_train[y_train == class_id]
        if len(class_vectors) == 0:
            continue
        proto = class_vectors.mean(axis=0)
        proto = proto / max(np.linalg.norm(proto), 1e-12)
        prototypes[class_id] = proto
        valid[class_id] = True
    scores = x_test @ prototypes.T
    scores[:, ~valid] = -np.inf
    return scores


def normalize_scores(scores):
    scores = scores.astype(np.float32)
    finite = np.isfinite(scores)
    if not finite.any():
        return scores
    mean = scores[finite].mean()
    std = scores[finite].std()
    return (scores - mean) / max(std, 1e-6)


def leave_one_out_eval(embeddings, labels, classifier_kind, output_dir, result_prefix=None, svm_c=1.0):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    classes = encoder.classes_

    y_true = []
    y_pred = []
    score_rows = []

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(embeddings):
        if classifier_kind in {"proto", "svm_proto"}:
            proto_full_scores = prototype_scores(embeddings[train_idx], y[train_idx], embeddings[test_idx], len(classes))[0]
            if classifier_kind == "proto":
                full_scores = proto_full_scores
            else:
                clf = make_classifier("svm", svm_c=svm_c)
                clf.fit(embeddings[train_idx], y[train_idx])
                svm_scores = clf.decision_function(embeddings[test_idx])[0]
                svm_full_scores = np.full(len(classes), -np.inf, dtype=np.float32)
                svm_full_scores[clf[-1].classes_] = svm_scores
                full_scores = 0.75 * normalize_scores(svm_full_scores) + 0.25 * normalize_scores(proto_full_scores)
            pred = int(np.argmax(full_scores))
            y_true.append(y[test_idx][0])
            y_pred.append(pred)
            score_rows.append(full_scores)
            continue

        clf = make_classifier(classifier_kind, svm_c=svm_c)
        clf.fit(embeddings[train_idx], y[train_idx])
        pred = clf.predict(embeddings[test_idx])[0]
        y_true.append(y[test_idx][0])
        y_pred.append(pred)

        estimator = clf[-1]
        if hasattr(estimator, "decision_function"):
            scores = clf.decision_function(embeddings[test_idx])[0]
            score_class_ids = estimator.classes_
        elif hasattr(estimator, "predict_proba"):
            scores = clf.predict_proba(embeddings[test_idx])[0]
            score_class_ids = estimator.classes_
        else:
            distances, neighbors = estimator.kneighbors(clf[:-1].transform(embeddings[test_idx]), n_neighbors=min(5, len(train_idx)))
            scores_map = {}
            for dist, neighbor_pos in zip(distances[0], neighbors[0]):
                class_id = y[train_idx][neighbor_pos]
                scores_map[class_id] = max(scores_map.get(class_id, -np.inf), -float(dist))
            score_class_ids = np.array(sorted(scores_map))
            scores = np.array([scores_map[c] for c in score_class_ids])

        full_scores = np.full(len(classes), -np.inf, dtype=np.float32)
        full_scores[score_class_ids] = scores
        score_rows.append(full_scores)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = np.vstack(score_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "classifier": classifier_kind,
        "samples": int(len(y_true)),
        "classes": int(len(classes)),
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": float(top_k_accuracy_score(y_true, scores, k=min(5, len(classes)), labels=np.arange(len(classes)))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    result_prefix = result_prefix or classifier_kind
    with open(output_dir / f"metrics_{result_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report = classification_report(y_true, y_pred, labels=np.arange(len(classes)), target_names=classes, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / f"classification_report_{result_prefix}.csv", encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(output_dir / f"confusion_matrix_{result_prefix}.csv", encoding="utf-8-sig")

    predictions = pd.DataFrame(
        {
            "true_class": encoder.inverse_transform(y_true),
            "predicted_class": encoder.inverse_transform(y_pred),
            "correct": y_true == y_pred,
        }
    )
    top5 = np.argsort(-scores, axis=1)[:, : min(5, len(classes))]
    for i in range(top5.shape[1]):
        predictions[f"top{i + 1}"] = classes[top5[:, i]]
    predictions.to_csv(output_dir / f"loo_predictions_{result_prefix}.csv", index=False, encoding="utf-8-sig")

    return metrics


def write_summary(df, output_dir):
    usable = df[df["usable_for_training"]]
    class_counts = usable.groupby("class_name").size().sort_values()
    duplicate_groups = df[df["duplicate_group_size"] > 1].sort_values(["sha256", "class_name", "file_name"])

    summary = {
        "raw_images": int(len(df)),
        "usable_images": int(len(usable)),
        "classes_raw": int(df["class_name"].nunique()),
        "classes_usable": int(usable["class_name"].nunique()),
        "min_images_per_class_usable": int(class_counts.min()),
        "max_images_per_class_usable": int(class_counts.max()),
        "avg_images_per_class_usable": float(class_counts.mean()),
        "duplicate_image_rows": int((df["duplicate_group_size"] > 1).sum()),
        "cross_class_duplicate_rows": int(df["cross_class_duplicate"].sum()),
        "same_class_duplicate_extras_excluded": int(df["same_class_duplicate_extra"].sum()),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    class_counts.to_csv(output_dir / "usable_class_counts.csv", encoding="utf-8-sig", header=["count"])
    duplicate_groups.to_csv(output_dir / "duplicate_groups.csv", index=False, encoding="utf-8-sig")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Bases de données photos Dr Taleb")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "vit_b_16", "dinov2_small", "dinov2_base"])
    parser.add_argument("--classifier", default="svm", choices=["svm", "svm_rbf", "knn", "proto", "svm_proto"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tta-crops", type=int, default=1)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--keep-cross-class-duplicates", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    metadata_path = output / "metadata_clean.csv"
    tta_suffix = f"_tta{args.tta_crops}" if args.tta_crops > 1 else ""
    c_suffix = f"_c{args.svm_c:g}".replace(".", "p") if args.classifier in {"svm", "svm_proto"} else ""
    result_prefix = f"{args.model}{tta_suffix}_{args.classifier}{c_suffix}"
    embeddings_path = output / f"embeddings_{args.model}{tta_suffix}.npz"

    df = scan_dataset(args.dataset, metadata_path, keep_cross_class_duplicates=args.keep_cross_class_duplicates)
    summary = write_summary(df, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.metadata_only:
        return

    if embeddings_path.exists():
        cached = np.load(embeddings_path, allow_pickle=True)
        embeddings = cached["embeddings"]
        labels = cached["labels"]
        print(f"Loaded cached embeddings: {embeddings_path}")
    else:
        embeddings, labels, _ = extract_embeddings(df, args.model, args.batch_size, embeddings_path, tta_crops=args.tta_crops)
    metrics = leave_one_out_eval(embeddings, labels, args.classifier, output, result_prefix=result_prefix, svm_c=args.svm_c)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

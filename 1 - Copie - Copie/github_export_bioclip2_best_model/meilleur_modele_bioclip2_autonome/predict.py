import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageFile

import open_clip


ImageFile.LOAD_TRUNCATED_IMAGES = True


def binomial_name(text):
    cleaned = re.sub(r"\([^)]*\)", " ", text)
    parts = re.findall(r"[A-Z]?[a-zA-Z]+(?:-[a-zA-Z]+)?", cleaned)
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return re.sub(r"\s+", " ", text).strip()


def load_classes(path):
    class_names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            _index, name = line.split("\t", 1)
            class_names.append(name.strip())
        else:
            class_names.append(line)
    return class_names


@torch.inference_mode()
def encode_text(model, tokenizer, class_names, prompt_templates, device):
    all_features = []
    for template in prompt_templates:
        prompts = [
            template.format(name=name, binomial=binomial_name(name))
            for name in class_names
        ]
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features)
    features = torch.stack(all_features).mean(dim=0)
    return features / features.norm(dim=-1, keepdim=True)


@torch.inference_mode()
def predict(image_path, top_k):
    base_dir = Path(__file__).resolve().parent
    config = json.loads((base_dir / "model_config.json").read_text(encoding="utf-8"))
    class_names = load_classes(base_dir / "classes.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _train_preprocess, preprocess = open_clip.create_model_and_transforms(
        config["arch"],
        pretrained=config.get("pretrained"),
    )
    tokenizer = open_clip.get_tokenizer(config["arch"])
    model.eval().to(device)

    text_features = encode_text(
        model,
        tokenizer,
        class_names,
        config["prompt_templates"],
        device,
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    logits = 100.0 * image_features @ text_features.T
    probabilities = logits.softmax(dim=-1)[0]
    scores, indices = probabilities.topk(k=min(top_k, len(class_names)))

    return [
        {
            "rank": rank,
            "class_index": int(index),
            "class_name": class_names[int(index)],
            "score": float(score),
        }
        for rank, (score, index) in enumerate(zip(scores.cpu(), indices.cpu()), start=1)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prediction avec BioCLIP 2 zero-shot sur les 138 classes Dr Taleb."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(Path("example_images") / "abutilon_theophrasti_example.JPG"),
        help="Chemin vers une image a classifier.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de predictions a afficher.")
    return parser.parse_args()


def main():
    args = parse_args()
    results = predict(args.image, args.top_k)
    print(json.dumps({"image": args.image, "predictions": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

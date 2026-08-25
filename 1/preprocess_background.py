import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def safe_name(name):
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in str(name))
    return out.strip().rstrip(".")


def grabcut_mask(rgb, work_size=384):
    h, w = rgb.shape[:2]
    if min(h, w) < 20:
        return np.ones((h, w), dtype=np.uint8)

    scale = min(1.0, float(work_size) / max(h, w))
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb.copy()
    sh, sw = small.shape[:2]

    mask = np.zeros((sh, sw), np.uint8)
    margin_x = max(2, int(sw * 0.04))
    margin_y = max(2, int(sh * 0.04))
    rect = (margin_x, margin_y, max(1, sw - 2 * margin_x), max(1, sh - 2 * margin_y))

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        mask = np.ones((sh, sw), dtype=np.uint8)

    if scale < 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Keep the largest connected foreground component plus nearby fine details.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        largest_area = areas.max()
        keep = labels == largest
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= max(50, 0.08 * largest_area):
                keep |= labels == label_id
        mask = keep.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def bbox_from_mask(mask, pad_ratio=0.08):
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, w, h
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    pad = int(max(x2 - x1, y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return x1, y1, x2, y2


def resize_max_side(rgb, max_side):
    if max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        return rgb
    return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def process_image(src, dst, mode, work_size=384, max_side=1024):
    if dst.exists():
        return None
    image = Image.open(src).convert("RGB")
    rgb = np.array(image)
    mask = grabcut_mask(rgb, work_size=work_size)

    if mode == "white":
        out = np.full_like(rgb, 255)
        out[mask > 0] = rgb[mask > 0]
    elif mode == "blur":
        blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=18, sigmaY=18)
        out = blur
        out[mask > 0] = rgb[mask > 0]
    elif mode == "white_crop":
        out = np.full_like(rgb, 255)
        out[mask > 0] = rgb[mask > 0]
        x1, y1, x2, y2 = bbox_from_mask(mask)
        out = out[y1:y2, x1:x2]
    else:
        raise ValueError("mode must be: white, blur, white_crop")

    dst.parent.mkdir(parents=True, exist_ok=True)
    out = resize_max_side(out, max_side)
    Image.fromarray(out).save(dst, quality=92)
    return int(mask.sum()), int(mask.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="outputs/metadata_clean.csv")
    parser.add_argument("--output", default="preprocessed/bg_white_crop")
    parser.add_argument("--mode", default="white_crop", choices=["white", "blur", "white_crop"])
    parser.add_argument("--work-size", type=int, default=384)
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    if "usable_for_training" in df.columns:
        df = df[df["usable_for_training"].astype(str).str.lower().isin(["true", "1"])]

    out_root = Path(args.output)
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        class_name = row["class_name"]
        src = Path(row["image_path"])
        dst = out_root / safe_name(class_name) / f"{i:05d}_{safe_name(src.stem)}.jpg"
        result = process_image(src, dst, args.mode, work_size=args.work_size, max_side=args.max_side)
        if result is None:
            fg_pixels, total_pixels = 0, 0
        else:
            fg_pixels, total_pixels = result
        rows.append(
            {
                "source_path": str(src),
                "output_path": str(dst),
                "class_name": class_name,
                "foreground_ratio": fg_pixels / max(total_pixels, 1) if total_pixels else "",
            }
        )
        if (i + 1) % 50 == 0:
            print(f"processed {i + 1}/{len(df)}", flush=True)

    ratios = [r["foreground_ratio"] for r in rows if isinstance(r["foreground_ratio"], (int, float))]
    summary = {
        "mode": args.mode,
        "images": len(rows),
        "output": str(out_root),
        "mean_foreground_ratio": float(np.mean(ratios)) if ratios else None,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "preprocess_metadata.csv", index=False, encoding="utf-8-sig")
    with open(out_root / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

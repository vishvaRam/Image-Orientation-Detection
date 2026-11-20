import time
from pathlib import Path

from paddleocr import DocImgOrientationClassification
from PIL import Image, ImageOps

# --- Configuration ---
ROOT_DIR = Path("Data")
OUTPUT_DIR = Path("output_corrected")
BATCH_SIZE = 8
CONFIDENCE_THRESHOLD = 0.70  # Skip rotation if prediction is uncertain
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_image_paths(root_dir: Path) -> list[Path]:
    return sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def correct_orientation(img_path: Path, angle: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as img:
        # Normalize any existing EXIF orientation metadata first
        img = ImageOps.exif_transpose(img)
        if angle in (90, 180, 270):
            # PaddleX reports CW rotation of the document; PIL.rotate is CCW
            img = img.rotate(angle, expand=True)
        img.save(out_path)


def main():
    total_start = time.perf_counter()

    image_paths = get_image_paths(ROOT_DIR)
    if not image_paths:
        print(f"No images found in '{ROOT_DIR}'.")
        return

    print(f"Found {len(image_paths)} images in '{ROOT_DIR}'. Loading model...")
    load_start = time.perf_counter()
    model = DocImgOrientationClassification(
        model_name="PP-LCNet_x1_0_doc_ori",
        engine="onnxruntime",
    )
    print(f"Model loaded in {time.perf_counter() - load_start:.2f}s\n")

    infer_start = time.perf_counter()
    rotated_count = 0

    # Process in batches for faster ONNX Runtime throughput
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        batch_str_paths = [str(p) for p in batch_paths]

        batch_t0 = time.perf_counter()
        results = model.predict(batch_str_paths, batch_size=len(batch_str_paths))

        for img_path, res in zip(batch_paths, results):
            # Handle PaddleX Result object dict access
            res_dict = res["res"] if "res" in res else res
            angle = int(res_dict["label_names"][0])
            score = float(res_dict["scores"][0])

            rel_path = img_path.relative_to(ROOT_DIR)
            out_path = OUTPUT_DIR / rel_path

            applied_angle = angle if (angle != 0 and score >= CONFIDENCE_THRESHOLD) else 0
            correct_orientation(img_path, applied_angle, out_path)

            if applied_angle != 0:
                rotated_count += 1

            print(
                f"[{rel_path}] pred={angle}° (conf={score:.3f}) -> "
                f"applied={applied_angle}° | saved to '{out_path}'"
            )

        batch_elapsed = time.perf_counter() - batch_t0
        print(f"--- Batch {i // BATCH_SIZE + 1} done in {batch_elapsed:.3f}s ---\n")

    total_elapsed = time.perf_counter() - total_start
    infer_elapsed = time.perf_counter() - infer_start

    print("=" * 50)
    print(f"Total images processed : {len(image_paths)}")
    print(f"Images rotated         : {rotated_count}")
    print(f"Inference + I/O time   : {infer_elapsed:.2f}s ({infer_elapsed / len(image_paths):.3f}s/img)")
    print(f"Total script time      : {total_elapsed:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()

from pathlib import Path
import shutil
import yaml

# Original dataset
SOURCE = Path(
    r"C:\Users\AMRISH A\Downloads\Cac_Segmentation27.v3i.yolov8"
)

# New cleaned dataset
DEST = Path(
    r"C:\Users\AMRISH A\Downloads\crater_rock_seg"
)

# Original IDs -> new IDs
CLASS_MAP = {
    4: 0,  # Crater
    5: 1   # Rock
}

SPLITS = ["train", "valid", "test"]

def clean_dataset():
    total_counts = {"Crater": 0, "Rock": 0}

    for split in SPLITS:
        image_dir = SOURCE / split / "images"
        label_dir = SOURCE / split / "labels"

        output_images = DEST / split / "images"
        output_labels = DEST / split / "labels"

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image folder: {image_dir}")

        images = [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in
            [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ]

        print(f"\nProcessing {split}: {len(images)} images")

        for image in images:
            source_label = label_dir / f"{image.stem}.txt"
            destination_image = output_images / image.name
            destination_label = output_labels / f"{image.stem}.txt"

            # Copy image without changing original
            shutil.copy2(image, destination_image)

            cleaned_lines = []

            if source_label.exists():
                for line in source_label.read_text().splitlines():
                    parts = line.strip().split()

                    if len(parts) < 7:
                        continue

                    try:
                        class_id = int(float(parts[0]))
                        coordinates = [float(x) for x in parts[1:]]
                    except ValueError:
                        continue

                    if class_id not in CLASS_MAP:
                        continue

                    # YOLO segmentation polygons need at least 3 points
                    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
                        continue

                    if not all(0 <= x <= 1 for x in coordinates):
                        continue

                    new_id = CLASS_MAP[class_id]
                    cleaned_lines.append(
                        str(new_id) + " " +
                        " ".join(str(x) for x in coordinates)
                    )

                    if new_id == 0:
                        total_counts["Crater"] += 1
                    else:
                        total_counts["Rock"] += 1

            destination_label.write_text(
                "\n".join(cleaned_lines)
                + ("\n" if cleaned_lines else "")
            )

        print(f"Completed {split}")

    # Create dataset YAML
    data = {
        "path": str(DEST).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": {
            0: "Crater",
            1: "Rock"
        }
    }

    with open(DEST / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print("\n========== CLEANING COMPLETE ==========")
    print("Crater annotations:", total_counts["Crater"])
    print("Rock annotations:", total_counts["Rock"])
    print("Output:", DEST)
    print("YAML:", DEST / "data.yaml")


if __name__ == "__main__":
    clean_dataset()

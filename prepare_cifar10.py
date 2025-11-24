from pathlib import Path
from torchvision.datasets import CIFAR10

# Root of this script
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMG_ROOT = DATA_DIR / "images"


def prepare_cifar10(max_per_class: int = 2000) -> None:
    """
    Download CIFAR-10 (if needed), then extract up to max_per_class images
    for class 0 (airplane) and class 1 (automobile), and save them as:

        data/images/class0/00000.png, ...
        data/images/class1/00000.png, ...

    This matches the folder structure expected by your CPC project.
    """
    print("=== Preparing CIFAR-10 images ===")
    
    # Where torchvision will store the raw CIFAR files
    raw_dir = DATA_DIR / "cifar10_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Only download if not already present
    ds = CIFAR10(
        root=str(raw_dir),
        download=not (raw_dir / "cifar-10-batches-py").exists(),
        train=True,
    )

    # Make sure output dirs exist
    (IMG_ROOT / "class0").mkdir(parents=True, exist_ok=True)
    (IMG_ROOT / "class1").mkdir(parents=True, exist_ok=True)

    count0 = 0
    count1 = 0

    for img, label in ds:
        # Keep only labels 0 (airplane) and 1 (automobile)
        if label not in (0, 1):
            continue

        if label == 0:
            cls = "class0"
            idx = count0
        else:
            cls = "class1"
            idx = count1

        out_dir = IMG_ROOT / cls
        out_path = out_dir / f"{idx:05d}.png"

        # CIFAR10 returns a PIL.Image by default -> directly save
        img.save(out_path)

        if cls == "class0":
            count0 += 1
        else:
            count1 += 1

        # Stop once we have enough for both classes
        if count0 >= max_per_class and count1 >= max_per_class:
            break

    print(f"Saved {count0} images to {IMG_ROOT / 'class0'}")
    print(f"Saved {count1} images to {IMG_ROOT / 'class1'}")


if __name__ == "__main__":
    print("Root:", ROOT)
    # Ensure base data dir exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepare_cifar10(max_per_class=2000)
    print("Done. Data ready under:", DATA_DIR)

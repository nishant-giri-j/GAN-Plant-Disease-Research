import os
import argparse
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

def resize_images(input_dir, output_dir, size=256):
    """
    Resizes images from raw folder and saves them to processed.
    Structure: data/raw/Disease_Name -> data/processed/Disease_Name
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = T.Compose([
        T.Resize((size, size)),
        T.CenterCrop(size), # Ensures square aspect ratio
    ])

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Found classes: {classes}")

    for cls in classes:
        src_path = os.path.join(input_dir, cls)
        dst_path = os.path.join(output_dir, cls)
        os.makedirs(dst_path, exist_ok=True)

        files = os.listdir(src_path)
        print(f"Processing {cls} ({len(files)} images)...")

        for fname in tqdm(files):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(os.path.join(src_path, fname)).convert("RGB")
                    img = transform(img)
                    img.save(os.path.join(dst_path, fname))
                except Exception as e:
                    print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    # Default Paths matching your structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    resize_images(RAW_DIR, PROC_DIR)
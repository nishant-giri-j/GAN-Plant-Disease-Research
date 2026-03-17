import os
import subprocess
import glob
import sys
import torch
import torch.nn as nn
import json
import re
import shutil
from torchvision.utils import save_image

# ==========================================
# ### STRATEGY CONFIGURATION ###
# ==========================================
TARGET_CLASS = "Septoria"
GPUS = "1"
DCGAN_EPOCHS = "200"
FASTGAN_ITER = "30000"
STD_GEN_COUNT = 1000
SCALED_GEN_COUNT = 5000

# USE ABSOLUTE PATHS TO PREVENT WINDOWS CONFUSION
BASE_DIR = os.path.abspath(os.getcwd())
DATA_PATH_TARGET = os.path.join(BASE_DIR, "data", "processed", TARGET_CLASS)
DATA_PATH_ROOT = os.path.join(BASE_DIR, "data", "processed")
FASTGAN_DATA_ROOT = os.path.join(BASE_DIR, "data", "fastgan_input")
CHECKPOINT_ROOT = os.path.join(BASE_DIR, "models", "checkpoints")
SYNTHETIC_ROOT = os.path.join(BASE_DIR, "data", "synthetic")
RESULTS_FILE = "research_results.json"
FASTGAN_REPO_DIR = os.path.join(BASE_DIR, "models", "FastGAN-pytorch")

os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

def run_command(cmd, capture=False):
    cmd_str = ' '.join(cmd) if isinstance(cmd, list) else cmd
    print(f"Running: {cmd_str}")
    if capture:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {e}")
            return ""
    else:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {e}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def update_results_file(entry_name, metric, value):
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    if entry_name not in data: data[entry_name] = {}
    data[entry_name][metric] = float(value)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f">>> Logged {metric.upper()} for {entry_name}: {value}")

def setup_fastgan():
    if os.path.exists(FASTGAN_REPO_DIR):
        print(f"\n---> FastGAN Repository found at: {FASTGAN_REPO_DIR}")
    else:
        print(f"\n---> WARNING: FastGAN folder not found at: {FASTGAN_REPO_DIR}")

def prepare_fastgan_data():
    """
    Robust data copying with verification.
    """
    print("\n---> PREPARING FASTGAN DATA STRUCTURE...")
    dest_dir = os.path.join(FASTGAN_DATA_ROOT, TARGET_CLASS)
    
    # 1. Check if files already exist
    if os.path.exists(dest_dir):
        existing_count = len([f for f in os.listdir(dest_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if existing_count > 0:
            print(f"---> Data already exists ({existing_count} images). Skipping copy.")
            return dest_dir

    # 1. Clean old folder to ensure fresh copy
    if os.path.exists(FASTGAN_DATA_ROOT):
        try:
            shutil.rmtree(FASTGAN_DATA_ROOT)
        except Exception as e:
            print(f"Warning: Could not clean folder: {e}")

    os.makedirs(dest_dir, exist_ok=True)
    
    # 2. Find images manually using glob for safety
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    src_files = []
    for t in types:
        src_files.extend(glob.glob(os.path.join(DATA_PATH_TARGET, t)))
    
    # De-duplicate files (Windows case insensitivity handling)
    src_files = list(set(src_files))
    
    print(f"Found {len(src_files)} unique images in {DATA_PATH_TARGET}")
    
    if len(src_files) == 0:
        print("CRITICAL ERROR: No images found! Check your data/processed/Septoria folder.")
        return None

    # 3. Copy
    print(f"Copying {len(src_files)} images to {dest_dir}...")
    for f in src_files:
        try:
            shutil.copy2(f, dest_dir)
        except:
            pass # Skip errors on individual files
        
    # 4. Verify
    final_count = len(os.listdir(dest_dir))
    print(f"Verification: {final_count} images successfully copied to {dest_dir}")
    
    if final_count == 0:
        print("CRITICAL ERROR: Copy failed.")
        return None
        
    return dest_dir

def train_model(gan_name):
    print(f"\n---> TRAINING: {gan_name}")
    
    if gan_name == "dcgan_diffaug":
        out_dir = os.path.join(CHECKPOINT_ROOT, "dcgan_diffaug")
        script = os.path.join(BASE_DIR, "models", "dcgan_diffaug", "train.py")
        ensure_dir(out_dir)
        # Check if DCGAN model already exists
        final_ckpt = os.path.join(out_dir, "generator.pth")
        if os.path.exists(final_ckpt):
            print(f"---> Found existing DCGAN model at {final_ckpt}. Skipping training.")
            return

        cmd = [sys.executable, script, "--dataroot", DATA_PATH_ROOT, "--outdir", out_dir, "--ngpu", "1", "--cuda", "--niter", DCGAN_EPOCHS]
        run_command(cmd)

    elif gan_name == "fastgan":
        setup_fastgan()
        
        training_path = prepare_fastgan_data()
        if not training_path: return 

        script = os.path.join(FASTGAN_REPO_DIR, "train.py")
        
        if not os.path.exists(script):
            print(f"CRITICAL ERROR: Cannot find train.py at {script}")
            return

        # FIX: Check if model already exists to avoid re-training
        final_ckpt = os.path.join(CHECKPOINT_ROOT, "fastgan", f"{FASTGAN_ITER}.pth")
        if os.path.exists(final_ckpt):
            print(f"---> Found existing model at {final_ckpt}. Skipping training.")
            return

        env = os.environ.copy()
        env["PYTHONPATH"] = FASTGAN_REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")

        # COMMAND TO TRAIN
        # FIX: --workers 0 is CRITICAL for Windows stability
        cmd = [
            sys.executable, script, 
            "--path", training_path, 
            "--iter", FASTGAN_ITER, 
            "--batch_size", "8", 
            "--im_size", "256", 
            "--output_path", os.path.join(CHECKPOINT_ROOT, "fastgan"),
            "--name", "septoria_fast",
            "--save_interval", "50",
            "--workers", "0"  # <--- THIS IS THE KEY FIX
        ]
        
        print(f"Executing FastGAN Training...")
        subprocess.run(cmd, env=env, check=False)

def generate_images(gan_name, count, folder_suffix=""):
    save_name = f"{gan_name}{folder_suffix}"
    print(f"\n---> GENERATING DATA: {save_name} (Count: {count})")
    out_dir = os.path.join(SYNTHETIC_ROOT, save_name, TARGET_CLASS)
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if gan_name == "dcgan_diffaug":
        model_path = os.path.join(CHECKPOINT_ROOT, "dcgan_diffaug", "generator.pth")
        if os.path.exists(model_path):
            class Generator(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.main = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True), nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True), nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True), nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh())
                def forward(self, input): return self.main(input)
            
            netG = Generator().to(device)
            netG.load_state_dict(torch.load(model_path, map_location=device))
            netG.eval()
            with torch.no_grad():
                for i in range(count):
                    noise = torch.randn(1, 100, 1, 1, device=device)
                    save_image(netG(noise), os.path.join(out_dir, f"{i}.png"), normalize=True)

    elif gan_name == "fastgan":
        if FASTGAN_REPO_DIR not in sys.path:
            sys.path.append(FASTGAN_REPO_DIR)
        
        try:
            from models import Generator
        except ImportError:
            print(f"CRITICAL ERROR: Could not import FastGAN models from {FASTGAN_REPO_DIR}")
            return

        # Correct path for FastGAN train results
        # Now points to models/checkpoints/fastgan
        ckpt_dir = os.path.join(CHECKPOINT_ROOT, "fastgan")
        
        if not os.path.exists(ckpt_dir):
            print(f"Error: Checkpoint folder not found at {ckpt_dir}")
            return

        # Filter for numeric checkpoints (e.g. 30000.pth) avoiding 'all_' or 'args.txt'
        pth_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth') and f[0].isdigit()]
        if not pth_files:
            print("Error: No Generator models found.")
            return

        latest_file = max(pth_files, key=lambda f: int(f.replace('.pth', '')))
        model_path = os.path.join(ckpt_dir, latest_file)
        print(f"Loading FastGAN Model: {model_path}")

        netG = Generator(ngf=64, nz=256, im_size=256, nc=3).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        # FastGAN saves dict with 'g' and 'd'. We need 'g'.
        if 'g' in checkpoint:
            state_dict = checkpoint['g']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        netG.load_state_dict(new_state_dict)
        netG.eval()

        print("Generating images...")
        batch_size = 8
        generated = 0
        with torch.no_grad():
            while generated < count:
                noise = torch.randn(batch_size, 256, device=device)
                fake_imgs = netG(noise)
                # FastGAN returns [high_res, small_res, part]. We want high_res (index 0).
                if isinstance(fake_imgs, list):
                    fake_imgs = fake_imgs[0]
                for i in range(fake_imgs.size(0)):
                    if generated >= count: break
                    img = (fake_imgs[i] + 1) / 2.0
                    save_image(img, os.path.join(out_dir, f"{generated}.jpg"))
                    generated += 1
        print("Done.")

def evaluate_model(entry_name, gan_folder_name=None):
    if gan_folder_name is None: gan_folder_name = entry_name
    print(f"\n---> EVALUATING: {entry_name}")
    real_path = os.path.join(DATA_PATH_ROOT, TARGET_CLASS)
    syn_folder = os.path.join(SYNTHETIC_ROOT, gan_folder_name, TARGET_CLASS)

    if os.path.exists(syn_folder) and len(os.listdir(syn_folder)) > 0:
        print(">>> Calculating FID...")
        output = run_command([sys.executable, "src/calc_fid.py", "--real_path", real_path, "--fake_path", syn_folder], capture=True)
        match = re.search(r"FID SCORE: ([\d\.]+)", output)
        if match: update_results_file(entry_name, "fid", match.group(1))

    print(">>> Training Classifier...")
    output = run_command([sys.executable, "src/train_classifier.py", "--gan_name", gan_folder_name, "--use_synthetic", "True"], capture=True)
    match = re.search(r"F1-Score = ([\d\.]+)", output)
    if match: update_results_file(entry_name, "f1", match.group(1))

def main():
    print("--- PREPARING DATA ---")
    run_command([sys.executable, "src/prepare_data.py"])

    while True:
        print("\n" + "="*50)
        print("   STRATEGIC RESEARCH PIPELINE: DCGAN vs FASTGAN")
        print("="*50)
        print("PHASE 1: THE FACE-OFF")
        print("1. Train & Test DCGAN (Baseline 1k)")
        print("2. Train & Test FastGAN (The 1-Day Challenger)")
        print("-" * 30)
        print("PHASE 2: THE VICTORY LAP (Scaling)")
        print("3. SCALE UP FASTGAN (FastGAN -> 5,000 Images)")
        print("-" * 30)
        print("4. Plot Final Graphs (Comparison + Scaling)")
        print("5. Exit")
        
        c = input("\nChoice: ").strip()

        if c == '1':
            train_model("dcgan_diffaug")
            generate_images("dcgan_diffaug", STD_GEN_COUNT)
            evaluate_model("dcgan_diffaug")
        elif c == '2':
            train_model("fastgan")
            generate_images("fastgan", STD_GEN_COUNT)
            evaluate_model("fastgan")
        elif c == '3':
            print("\n>>> STARTING VICTORY LAP: SCALING UP FASTGAN...")
            generate_images("fastgan", SCALED_GEN_COUNT, folder_suffix="_5k")
            evaluate_model("fastgan_5k", gan_folder_name="fastgan_5k")
        elif c == '4':
            print("\nEvaluating Baseline (Real Only) first...")
            output = run_command([sys.executable, "src/train_classifier.py", "--gan_name", "baseline", "--use_synthetic", "False"], capture=True)
            match = re.search(r"F1-Score = ([\d\.]+)", output)
            if match: update_results_file("baseline", "f1", match.group(1))
            print("Generating Graphs...")
            run_command([sys.executable, "src/plot_graphs.py"])
        elif c == '5': break

if __name__ == "__main__":
    main()
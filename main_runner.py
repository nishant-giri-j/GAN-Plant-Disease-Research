import os
import subprocess
import glob
import sys
import torch
import torch.nn as nn

# ==========================================
# ### USER CONFIGURATION (GPU MODE) ###
# ==========================================
TARGET_CLASS = "Septoria"       # The specific disease folder in data/processed/
GENERATE_COUNT = 1000           # Research Standard: 1000 images (Change to 5 for quick testing)
GPUS = "1"                      # Using 1 GPU

# Enable ALL GANs for full research comparison
DO_TRAIN = {
    "dcgan_diffaug": True,
    "stylegan2_ada": True,
    "projected_gan": True,
    "fastgan": True,
    "vision_aided": True
}

DO_GENERATE = {
    "dcgan_diffaug": True,
    "stylegan2_ada": True,
    "projected_gan": True,
    "fastgan": True,
    "vision_aided": True
}

DO_EVALUATE = True
# ==========================================

# --- PATH DEFINITIONS ---
BASE_DIR = os.getcwd()
DATA_PATH_TARGET = os.path.join(BASE_DIR, "data", "processed", TARGET_CLASS)
DATA_PATH_ROOT = os.path.join(BASE_DIR, "data", "processed")

CHECKPOINT_ROOT = os.path.join(BASE_DIR, "models", "checkpoints")
SYNTHETIC_ROOT = os.path.join(BASE_DIR, "data", "synthetic")

os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

# --- UTILITIES ---
def run_command(cmd):
    """Prints and executes a system command."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR executing command: {e}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def find_latest_stylegan_pkl(checkpoint_folder):
    """Finds the latest .pkl file in nested StyleGAN experiment folders."""
    if not os.path.exists(checkpoint_folder): return None
    subfolders = sorted([f for f in glob.glob(os.path.join(checkpoint_folder, "*")) if os.path.isdir(f)])
    for folder in reversed(subfolders):
        pkls = sorted(glob.glob(os.path.join(folder, "network-snapshot-*.pkl")))
        if pkls: return pkls[-1]
    return None

# ---------------------------------------------------------
# PHASE 1: TRAINING FUNCTIONS (GPU OPTIMIZED)
# ---------------------------------------------------------
def train_all_models():
    print("\n" + "="*40 + "\n      PHASE 1: TRAINING MODELS (GPU)\n" + "="*40)

    # --- 1. DCGAN + DiffAugment ---
    if DO_TRAIN["dcgan_diffaug"]:
        print("\n---> Training DCGAN + DiffAugment")
        out_dir = os.path.join(CHECKPOINT_ROOT, "dcgan_diffaug")
        script = os.path.join(BASE_DIR, "models", "dcgan_diffaug", "train.py")
        ensure_dir(out_dir)
        # Uses --ngpu 1 for CUDA acceleration
        cmd = [sys.executable, script, "--dataroot", DATA_PATH_ROOT, "--outdir", out_dir, "--ngpu", "1", "--cuda"]
        run_command(cmd)

    # --- 2. StyleGAN2-ADA ---
    if DO_TRAIN["stylegan2_ada"]:
        print("\n---> Training StyleGAN2-ADA")
        out_dir = os.path.join(CHECKPOINT_ROOT, "stylegan2_ada")
        script = os.path.join(BASE_DIR, "models", "stylegan2-ada-pytorch", "train.py")
        # snap=50 saves checkpoints often. kimg=1000 is standard for small data.
        cmd = [sys.executable, script, "--outdir", out_dir, "--data", DATA_PATH_TARGET, "--gpus", GPUS, "--kimg", "1000", "--snap", "50"]
        run_command(cmd)

    # --- 3. Projected GAN ---
    if DO_TRAIN["projected_gan"]:
        print("\n---> Training Projected GAN")
        out_dir = os.path.join(CHECKPOINT_ROOT, "projected_gan")
        script = os.path.join(BASE_DIR, "models", "projected_gan", "train.py")
        # Projected GAN converges very fast. 1000 kimg is plenty.
        cmd = [sys.executable, script, "--outdir", out_dir, "--data", DATA_PATH_TARGET, "--batch", "16", "--kimg", "1000"]
        run_command(cmd)

    # --- 4. FastGAN ---
    if DO_TRAIN["fastgan"]:
        print("\n---> Training FastGAN")
        out_dir = os.path.join(CHECKPOINT_ROOT, "fastgan")
        script = os.path.join(BASE_DIR, "models", "FastGAN-pytorch", "train.py")
        ensure_dir(out_dir)
        # 5000 iterations is standard for FastGAN on small datasets
        cmd = [sys.executable, script, "--path", DATA_PATH_TARGET, "--output_path", out_dir, "--iter", "5000", "--batch_size", "8"]
        run_command(cmd)

    # --- 5. Vision-Aided GAN ---
    if DO_TRAIN["vision_aided"]:
        print("\n---> Training Vision-Aided GAN")
        out_dir = os.path.join(CHECKPOINT_ROOT, "vision_aided")
        script = os.path.join(BASE_DIR, "models", "vision-aided-gan", "stylegan2", "train.py")
        if os.path.exists(script):
            # Uses StyleGAN2 backbone with extra computer vision loss
            cmd = [sys.executable, script, "--outdir", out_dir, "--data", DATA_PATH_TARGET, "--gpus", GPUS, "--kimg", "1000", "--cfg", "auto"]
            run_command(cmd)

# ---------------------------------------------------------
# PHASE 2: GENERATION FUNCTIONS (GPU OPTIMIZED)
# ---------------------------------------------------------
def generate_dcgan_internal(model_path, output_dir, count):
    print(f"Generating DCGAN images from {model_path}...")
    
    # Re-define Generator Architecture (Must match train.py)
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(64 * 8), nn.ReLU(True),
                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4), nn.ReLU(True),
                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2), nn.ReLU(True),
                nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        def forward(self, input): return self.main(input)

    # GPU GENERATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    
    try:
        netG.load_state_dict(torch.load(model_path, map_location=device))
        netG.eval()
        ensure_dir(output_dir)
        from torchvision.utils import save_image
        print(f"Generating {count} images on {device}...")
        
        with torch.no_grad():
            for i in range(count):
                noise = torch.randn(1, 100, 1, 1, device=device)
                save_image(netG(noise), os.path.join(output_dir, f"{i}.png"), normalize=True)
        print(f"Saved to {output_dir}")
    except Exception as e:
        print(f"DCGAN Gen Error: {e}")

def generate_all_images():
    print("\n" + "="*40 + "\n      PHASE 2: GENERATING DATA (GPU)\n" + "="*40)

    # 1. DCGAN
    if DO_GENERATE["dcgan_diffaug"]:
        model_path = os.path.join(CHECKPOINT_ROOT, "dcgan_diffaug", "generator.pth")
        out_dir = os.path.join(SYNTHETIC_ROOT, "dcgan_diffaug", TARGET_CLASS)
        if os.path.exists(model_path):
            generate_dcgan_internal(model_path, out_dir, GENERATE_COUNT)

    # 2. StyleGAN2
    if DO_GENERATE["stylegan2_ada"]:
        pkl = find_latest_stylegan_pkl(os.path.join(CHECKPOINT_ROOT, "stylegan2_ada"))
        if pkl:
            script = os.path.join(BASE_DIR, "models", "stylegan2-ada-pytorch", "generate.py")
            out_dir = os.path.join(SYNTHETIC_ROOT, "stylegan2_ada", TARGET_CLASS)
            # StyleGAN generate.py automatically uses GPU if available
            cmd = [sys.executable, script, "--outdir", out_dir, "--seeds", f"0-{GENERATE_COUNT-1}", "--network", pkl]
            run_command(cmd)

    # 3. Projected GAN
    if DO_GENERATE["projected_gan"]:
        pkl = find_latest_stylegan_pkl(os.path.join(CHECKPOINT_ROOT, "projected_gan"))
        if pkl:
            script = os.path.join(BASE_DIR, "models", "projected_gan", "gen_images.py")
            if not os.path.exists(script): script = os.path.join(BASE_DIR, "models", "projected_gan", "generate.py")
            out_dir = os.path.join(SYNTHETIC_ROOT, "projected_gan", TARGET_CLASS)
            cmd = [sys.executable, script, "--outdir", out_dir, "--seeds", f"0-{GENERATE_COUNT-1}", "--network", pkl]
            run_command(cmd)

    # 4. FastGAN
    if DO_GENERATE["fastgan"]:
        cp_dir = os.path.join(CHECKPOINT_ROOT, "fastgan")
        pths = sorted(glob.glob(os.path.join(cp_dir, "*.pth")))
        if pths:
            script = os.path.join(BASE_DIR, "models", "FastGAN-pytorch", "eval.py")
            out_dir = os.path.join(SYNTHETIC_ROOT, "fastgan", TARGET_CLASS)
            ensure_dir(out_dir)
            cmd = [sys.executable, script, "--path", DATA_PATH_TARGET, "--output_path", out_dir, "--checkpoint", pths[-1], "--num", str(GENERATE_COUNT)]
            run_command(cmd)

    # 5. Vision-Aided GAN
    if DO_GENERATE["vision_aided"]:
        pkl = find_latest_stylegan_pkl(os.path.join(CHECKPOINT_ROOT, "vision_aided"))
        if pkl:
            script = os.path.join(BASE_DIR, "models", "vision-aided-gan", "stylegan2", "generate.py")
            out_dir = os.path.join(SYNTHETIC_ROOT, "vision_aided", TARGET_CLASS)
            cmd = [sys.executable, script, "--outdir", out_dir, "--seeds", f"0-{GENERATE_COUNT-1}", "--network", pkl]
            run_command(cmd)

# ---------------------------------------------------------
# PHASE 3: EVALUATION
# ---------------------------------------------------------
def evaluate_pipeline():
    print("\n" + "="*40 + "\n      PHASE 3: EVALUATION\n" + "="*40)
    
    # 1. Baseline
    print("\n--- BASELINE EVALUATION ---")
    run_command([sys.executable, "src/train_classifier.py", "--gan_name", "baseline", "--use_synthetic", "False"])

    gan_list = ["dcgan_diffaug", "stylegan2_ada", "projected_gan", "fastgan", "vision_aided"]
    
    for gan in gan_list:
        syn_class_folder = os.path.join(SYNTHETIC_ROOT, gan, TARGET_CLASS)
        
        if os.path.exists(syn_class_folder) and len(os.listdir(syn_class_folder)) > 0:
            print(f"\n>>> EVALUATING: {gan}")
            
            # A. Calculate FID (Realism Score)
            real_target_path = os.path.join(DATA_PATH_ROOT, TARGET_CLASS)
            run_command([
                sys.executable, "src/calc_fid.py",
                "--real_path", real_target_path,
                "--fake_path", syn_class_folder
            ])
            
            # B. Train Classifier (Usefulness Score)
            run_command([
                sys.executable, "src/train_classifier.py",
                "--gan_name", gan,
                "--use_synthetic", "True"
            ])

def main():
    print("--- PREPARING DATA ---")
    run_command([sys.executable, "src/prepare_data.py"])
    
    train_all_models()
    generate_all_images()
    
    if DO_EVALUATE:
        evaluate_pipeline()

if __name__ == "__main__":
    main()
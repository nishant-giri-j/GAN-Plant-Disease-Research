import os
import subprocess
import glob
import sys
import torch
import torch.nn as nn

# ==========================================
# ### USER CONFIGURATION ###
# ==========================================
TARGET_CLASS = "Septoria"       # The folder name in data/processed/
GPUS = "1"                      # Number of GPUs

# --- MODE SELECTION ---
# "TEST"     = Fast check (Mins). Use to verify code works.
# "RESEARCH" = Long run (Days). Use for 50k dataset results.
MODE = "RESEARCH" 

if MODE == "TEST":
    print(">>> MODE: FAST TEST (Quality will be low)")
    GENERATE_COUNT = 10         
    STYLEGAN_KIMG = "20"        
    PROJECTED_KIMG = "20"       
    LIGHTWEIGHT_STEPS = "100"   
    DCGAN_EPOCHS = "5"          

elif MODE == "RESEARCH":
    print(">>> MODE: FULL RESEARCH (High Quality)")
    GENERATE_COUNT = 1000       
    STYLEGAN_KIMG = "2000"      # ~4 Days
    PROJECTED_KIMG = "1000"     # ~24 Hours
    LIGHTWEIGHT_STEPS = "150000" # ~15 Hours
    DCGAN_EPOCHS = "200"        # ~6 Hours

# ==========================================
#Path Definitions
BASE_DIR = os.getcwd()
DATA_PATH_TARGET = os.path.join(BASE_DIR, "data", "processed", TARGET_CLASS)
DATA_PATH_ROOT = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_ROOT = os.path.join(BASE_DIR, "models", "checkpoints")
SYNTHETIC_ROOT = os.path.join(BASE_DIR, "data", "synthetic")
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

# --- UTILITIES ---
def run_command(cmd):
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def find_latest_stylegan_pkl(checkpoint_folder):
    if not os.path.exists(checkpoint_folder): return None
    subfolders = sorted([f for f in glob.glob(os.path.join(checkpoint_folder, "*")) if os.path.isdir(f)])
    for folder in reversed(subfolders):
        pkls = sorted(glob.glob(os.path.join(folder, "network-snapshot-*.pkl")))
        if pkls: return pkls[-1]
    return None

# ---------------------------------------------------------
# TRAINING FUNCTIONS
# ---------------------------------------------------------
def train_model(gan_name):
    print(f"\n---> STARTING TRAINING: {gan_name}")
    
    if gan_name == "dcgan_diffaug":
        out_dir = os.path.join(CHECKPOINT_ROOT, "dcgan_diffaug")
        script = os.path.join(BASE_DIR, "models", "dcgan_diffaug", "train.py")
        ensure_dir(out_dir)
        cmd = [sys.executable, script, "--dataroot", DATA_PATH_ROOT, "--outdir", out_dir, 
               "--ngpu", "1", "--cuda", "--niter", DCGAN_EPOCHS]
        run_command(cmd)

    elif gan_name == "lightweight_gan":
        out_dir = os.path.join(CHECKPOINT_ROOT, "lightweight_gan")
        ensure_dir(out_dir)
        try:
            cmd = [
                "lightweight_gan",
                "--data", DATA_PATH_TARGET,
                "--results_dir", out_dir,
                "--name", "septoria_model",
                "--image-size", "256",
                "--batch-size", "16",
                "--gradient-accumulate-every", "4",
                "--num-train-steps", LIGHTWEIGHT_STEPS
            ]
            run_command(cmd)
        except Exception:
            print("! Error: 'lightweight_gan' not found. Run: pip install lightweight-gan")

    elif gan_name == "projected_gan":
        out_dir = os.path.join(CHECKPOINT_ROOT, "projected_gan")
        script = os.path.join(BASE_DIR, "models", "projected_gan", "train.py")
        cmd = [sys.executable, script, "--outdir", out_dir, "--data", DATA_PATH_TARGET, 
               "--batch", "16", "--kimg", PROJECTED_KIMG]
        run_command(cmd)

    elif gan_name == "stylegan2_ada":
        out_dir = os.path.join(CHECKPOINT_ROOT, "stylegan2_ada")
        script = os.path.join(BASE_DIR, "models", "stylegan2-ada-pytorch", "train.py")
        cmd = [sys.executable, script, "--outdir", out_dir, "--data", DATA_PATH_TARGET, 
               "--gpus", GPUS, "--kimg", STYLEGAN_KIMG, "--snap", "20"]
        run_command(cmd)

# ---------------------------------------------------------
# GENERATION FUNCTIONS
# ---------------------------------------------------------
def generate_images(gan_name):
    print(f"\n---> GENERATING IMAGES: {gan_name}")
    out_dir = os.path.join(SYNTHETIC_ROOT, gan_name, TARGET_CLASS)
    ensure_dir(out_dir)

    if gan_name == "dcgan_diffaug":
        model_path = os.path.join(CHECKPOINT_ROOT, "dcgan_diffaug", "generator.pth")
        if os.path.exists(model_path):
            # Internal Class Definition for DCGAN
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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            netG = Generator().to(device)
            netG.load_state_dict(torch.load(model_path, map_location=device))
            netG.eval()
            
            from torchvision.utils import save_image
            print(f"Generating {GENERATE_COUNT} images...")
            with torch.no_grad():
                for i in range(GENERATE_COUNT):
                    noise = torch.randn(1, 100, 1, 1, device=device)
                    save_image(netG(noise), os.path.join(out_dir, f"{i}.png"), normalize=True)

    elif gan_name == "lightweight_gan":
        cmd = [
            "lightweight_gan",
            "--load-from", os.path.join(CHECKPOINT_ROOT, "lightweight_gan", "septoria_model"),
            "--generate", "--num-image-tiles", "1",
            "--results_dir", out_dir, "--image-size", "256"
        ]
        # Run loop to match generate count (Simulated for CLI)
        print("Running generation (Note: CLI generates batches)")
        run_command(cmd)

    elif gan_name == "projected_gan":
        pkl = find_latest_stylegan_pkl(os.path.join(CHECKPOINT_ROOT, "projected_gan"))
        if pkl:
            script = os.path.join(BASE_DIR, "models", "projected_gan", "gen_images.py")
            if not os.path.exists(script): script = os.path.join(BASE_DIR, "models", "projected_gan", "generate.py")
            cmd = [sys.executable, script, "--outdir", out_dir, "--seeds", f"0-{GENERATE_COUNT-1}", "--network", pkl]
            run_command(cmd)

    elif gan_name == "stylegan2_ada":
        pkl = find_latest_stylegan_pkl(os.path.join(CHECKPOINT_ROOT, "stylegan2_ada"))
        if pkl:
            script = os.path.join(BASE_DIR, "models", "stylegan2-ada-pytorch", "generate.py")
            cmd = [sys.executable, script, "--outdir", out_dir, "--seeds", f"0-{GENERATE_COUNT-1}", "--network", pkl]
            run_command(cmd)

# ---------------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------------
def evaluate_model(gan_name):
    print(f"\n---> EVALUATING: {gan_name}")
    real_path = os.path.join(DATA_PATH_ROOT, TARGET_CLASS)
    syn_folder = os.path.join(SYNTHETIC_ROOT, gan_name, TARGET_CLASS)
    
    if not os.path.exists(syn_folder) or len(os.listdir(syn_folder)) == 0:
        print("No images found to evaluate.")
        return

    # 1. FID
    run_command([sys.executable, "src/calc_fid.py", "--real_path", real_path, "--fake_path", syn_folder])
    
    # 2. Classifier
    run_command([sys.executable, "src/train_classifier.py", "--gan_name", gan_name, "--use_synthetic", "True"])

# ---------------------------------------------------------
# MAIN INTERACTIVE LOOP
# ---------------------------------------------------------
def main():
    # Pre-check data
    print("--- CHECKING DATA ---")
    run_command([sys.executable, "src/prepare_data.py"])

    while True:
        print("\n" + "="*40)
        print("      GAN RESEARCH CONTROL PANEL")
        print("="*40)
        print("Select a model to run (Train -> Generate -> Evaluate):")
        print("1. DCGAN + DiffAugment   (Est: 6 hrs)")
        print("2. Lightweight GAN       (Est: 15 hrs)")
        print("3. Projected GAN         (Est: 24 hrs)")
        print("4. StyleGAN2-ADA         (Est: 4 Days)")
        print("-" * 20)
        print("5. Run Evaluation Only (All Models)")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()

        if choice == '1':
            target = "dcgan_diffaug"
            train_model(target)
            generate_images(target)
            evaluate_model(target)
        elif choice == '2':
            target = "lightweight_gan"
            train_model(target)
            generate_images(target)
            evaluate_model(target)
        elif choice == '3':
            target = "projected_gan"
            train_model(target)
            generate_images(target)
            evaluate_model(target)
        elif choice == '4':
            target = "stylegan2_ada"
            train_model(target)
            generate_images(target)
            evaluate_model(target)
        elif choice == '5':
            print("\nRunning Evaluation on all existing models...")
            # Baseline first
            run_command([sys.executable, "src/train_classifier.py", "--gan_name", "baseline", "--use_synthetic", "False"])
            for g in ["dcgan_diffaug", "lightweight_gan", "projected_gan", "stylegan2_ada"]:
                evaluate_model(g)
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
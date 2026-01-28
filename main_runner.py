import os
import subprocess
import glob
import sys
import torch
import torch.nn as nn
import json

# ==========================================
# ### RESEARCH PROBLEM CONFIGURATION ###
# ==========================================
TARGET_CLASS = "Septoria"       # The disease class to augment
GPUS = "1"                      # Number of GPUs available

# --- EXPERIMENTAL SETTINGS ---
# "TEST": Fast verification (1 hr). "RESEARCH": Full paper results (Days).
MODE = "RESEARCH" 

if MODE == "TEST":
    print(">>> MODE: FAST PROTOTYPING (Low Quality)")
    GENERATE_COUNT = 50         # Generate minimal images
    STYLEGAN_KIMG = "20"        
    PROJECTED_KIMG = "20"       
    LIGHTWEIGHT_STEPS = "100"   
    DCGAN_EPOCHS = "5"          

elif MODE == "RESEARCH":
    print(">>> MODE: FULL ACADEMIC STUDY (High Fidelity)")
    GENERATE_COUNT = 1000       # Standard sample size for FID
    # Timings tuned for 50k dataset size:
    STYLEGAN_KIMG = "2000"      # High Fidelity (Slowest)
    PROJECTED_KIMG = "1000"     # Efficient High Quality (Best Trade-off)
    LIGHTWEIGHT_STEPS = "150000" # Mobile/Edge Optimization Focus
    DCGAN_EPOCHS = "200"        # Baseline Comparison

# ==========================================
# --- PATH DEFINITIONS ---
BASE_DIR = os.getcwd()
DATA_PATH_TARGET = os.path.join(BASE_DIR, "data", "processed", TARGET_CLASS)
DATA_PATH_ROOT = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_ROOT = os.path.join(BASE_DIR, "models", "checkpoints")
SYNTHETIC_ROOT = os.path.join(BASE_DIR, "data", "synthetic")
RESULTS_LOG = os.path.join(BASE_DIR, "research_results.json")

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
# PHASE 1: TRAINING (ARCHITECTURAL COMPARISON)
# ---------------------------------------------------------
def train_model(gan_name):
    print(f"\n---> [RESEARCH PHASE 1] TRAINING ARCHITECTURE: {gan_name}")
    
    if gan_name == "dcgan_diffaug":
        # Baseline Model
        out_dir = os.path.join(CHECKPOINT_ROOT, "dcgan_diffaug")
        script = os.path.join(BASE_DIR, "models", "dcgan_diffaug", "train.py")
        ensure_dir(out_dir)
        cmd = [sys.executable, script, "--dataroot", DATA_PATH_ROOT, "--outdir", out_dir, 
               "--ngpu", "1", "--cuda", "--niter", DCGAN_EPOCHS]
        run_command(cmd)

    elif gan_name == "lightweight_gan":
        # Efficiency Model (Addressing "Biologically Restricted" compute)
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
        # High-Fidelity Model
        out_dir = os.path.join(CHECKPOINT_ROOT, "projected_gan")
        script = os.path.join(BASE_DIR, "models", "projected_gan", "train.py")
        cmd = [sys.executable, script, "--outdir", out_dir, "--data", DATA_PATH_TARGET, 
               "--batch", "16", "--kimg", PROJECTED_KIMG]
        run_command(cmd)

    elif gan_name == "stylegan2_ada":
        # State-of-the-Art Benchmark
        out_dir = os.path.join(CHECKPOINT_ROOT, "stylegan2_ada")
        script = os.path.join(BASE_DIR, "models", "stylegan2-ada-pytorch", "train.py")
        cmd = [sys.executable, script, "--outdir", out_dir, "--data", DATA_PATH_TARGET, 
               "--gpus", GPUS, "--kimg", STYLEGAN_KIMG, "--snap", "20"]
        run_command(cmd)

# ---------------------------------------------------------
# PHASE 2: GENERATION (SYNTHETIC INTEGRATION)
# ---------------------------------------------------------
def generate_images(gan_name):
    print(f"\n---> [RESEARCH PHASE 2] GENERATING DATA: {gan_name}")
    out_dir = os.path.join(SYNTHETIC_ROOT, gan_name, TARGET_CLASS)
    ensure_dir(out_dir)

    if gan_name == "dcgan_diffaug":
        model_path = os.path.join(CHECKPOINT_ROOT, "dcgan_diffaug", "generator.pth")
        if os.path.exists(model_path):
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
        print("Running generation...")
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
# PHASE 3: EVALUATION (HYPOTHESIS TESTING)
# ---------------------------------------------------------
def evaluate_model(gan_name):
    print(f"\n---> [RESEARCH PHASE 3] TESTING HYPOTHESIS: {gan_name}")
    real_path = os.path.join(DATA_PATH_ROOT, TARGET_CLASS)
    syn_folder = os.path.join(SYNTHETIC_ROOT, gan_name, TARGET_CLASS)
    
    if not os.path.exists(syn_folder) or len(os.listdir(syn_folder)) == 0:
        print("No synthetic images found. Skipping evaluation.")
        return

    # 1. FID Score (Quantifying Image Quality)
    print(">>> Calculating FID (Lower is Better)...")
    run_command([sys.executable, "src/calc_fid.py", "--real_path", real_path, "--fake_path", syn_folder])
    
    # 2. Classifier Improvement (Quantifying Utility)
    print(">>> Training Classifier (Higher F1 is Better)...")
    # Note: 'train_classifier.py' prints the final F1 score improvement
    run_command([sys.executable, "src/train_classifier.py", "--gan_name", gan_name, "--use_synthetic", "True"])

# ---------------------------------------------------------
# MAIN CONTROL PANEL
# ---------------------------------------------------------
def main():
    print("--- PREPARING DATASETS ---")
    run_command([sys.executable, "src/prepare_data.py"])

    while True:
        print("\n" + "="*50)
        print("   RESEARCH PIPELINE CONTROLLER: SEPTORIA GAN")
        print("="*50)
        print("Current Goal: Evaluate Synthetic Data Integration")
        print("\nSelect Experiment:")
        print("1. Train Baseline (DCGAN)          [Est: 6 hrs]")
        print("2. Train Efficient (Lightweight)   [Est: 15 hrs] *Recommended*")
        print("3. Train High-Fidelity (Projected) [Est: 24 hrs]")
        print("4. Train State-of-Art (StyleGAN2)  [Est: 4 Days]")
        print("-" * 30)
        print("5. Run Full Evaluation Protocol (All Models)")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()

        if choice in ['1', '2', '3', '4']:
            model_map = {'1': "dcgan_diffaug", '2': "lightweight_gan", '3': "projected_gan", '4': "stylegan2_ada"}
            target = model_map[choice]
            
            # Execute Full Pipeline for selected model
            train_model(target)
            generate_images(target)
            evaluate_model(target)
            
        elif choice == '5':
            print("\n>>> STARTING COMPARATIVE EVALUATION...")
            # Step A: Establish Pure Baseline (No Synthetic Data)
            print("\n[BENCHMARK] Training Real-Data Only Baseline...")
            run_command([sys.executable, "src/train_classifier.py", "--gan_name", "baseline", "--use_synthetic", "False"])
            
            # Step B: Evaluate Each GAN
            for g in ["dcgan_diffaug", "lightweight_gan", "projected_gan", "stylegan2_ada"]:
                evaluate_model(g)
                
        elif choice == '6':
            print("Exiting pipeline.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
import os
import torch
import argparse
from pytorch_fid import fid_score

def calculate_fid(real_path, fake_path):
    print(f"\n[FID] Calculating Distance between:")
    print(f" Real: {real_path}")
    print(f" Fake: {fake_path}")

    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print("Error: One of the paths does not exist.")
        return 999.0

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fid_value = fid_score.calculate_fid_given_paths(
            [real_path, fake_path],
            batch_size=50,
            device=device,
            dims=2048
        )
        print(f"FID SCORE: {fid_value:.4f}")
        return fid_value
    except Exception as e:
        print(f"Failed to calculate FID: {e}")
        return 999.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, required=True)
    parser.add_argument("--fake_path", type=str, required=True)
    args = parser.parse_args()
    
    calculate_fid(args.real_path, args.fake_path)
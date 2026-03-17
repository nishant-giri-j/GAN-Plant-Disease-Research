import matplotlib.pyplot as plt
import json
import os

RESULTS_FILE = "research_results.json"

def plot_results():
    if not os.path.exists(RESULTS_FILE):
        print("No results file found.")
        return

    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    # ==========================================
    # CHART 1: FID SCORE (The Realism Test)
    # Compare DCGAN vs FastGAN. Lower is Better.
    # ==========================================
    # UPDATED: Changed 'lightweight_gan' to 'fastgan'
    gan_models = ["dcgan_diffaug", "fastgan"]
    gan_labels = ["DCGAN", "FastGAN (Ours)"]
    fids = []
    
    for m in gan_models:
        fids.append(data.get(m, {}).get("fid", 0))

    if any(fids):
        plt.figure(figsize=(8, 6))
        # Red (Bad) vs Green (Good)
        colors = ['#e74c3c', '#2ecc71'] 
        bars = plt.bar(gan_labels, fids, color=colors)
        
        plt.title("Visual Realism (FID Score)", fontsize=14, fontweight='bold')
        plt.ylabel("FID Score (Lower is Better)")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add numbers on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 1, 
                         f'{height:.1f}', ha='center', va='bottom', fontsize=12)
        
        plt.savefig("chart_1_fid_realism.png")
        print("Saved: chart_1_fid_realism.png")

    # ==========================================
    # CHART 2: F1 SCORE (The Utility Test)
    # Compare Baseline vs DCGAN vs FastGAN. Higher is Better.
    # ==========================================
    # UPDATED: Changed 'lightweight_gan' to 'fastgan'
    models_1 = ["baseline", "dcgan_diffaug", "fastgan"]
    labels_1 = ["Baseline (Real Only)", "Real + DCGAN", "Real + FastGAN"]
    f1s_1 = []
    
    for m in models_1:
        f1s_1.append(data.get(m, {}).get("f1", 0))

    if any(f1s_1):
        plt.figure(figsize=(10, 6))
        # Grey (Base) -> Red (DCGAN) -> Green (Winner)
        bars = plt.bar(labels_1, f1s_1, color=['gray', '#e74c3c', '#2ecc71'])
        
        plt.title("Classifier Accuracy (F1-Score)", fontsize=14, fontweight='bold')
        plt.ylabel("F1 Score (Higher is Better)")
        
        plt.ylim(0.0, 1.1) 
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                         f'{height:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.savefig("chart_2_f1_accuracy.png")
        print("Saved: chart_2_f1_accuracy.png")

    # ==========================================
    # CHART 3: SCALING ANALYSIS (The Victory Lap)
    # Compare 1k vs 5k.
    # ==========================================
    # UPDATED: Checking for 'fastgan_5k'
    if "fastgan_5k" in data:
        models_2 = ["baseline", "fastgan", "fastgan_5k"]
        labels_2 = ["Real Data Only", "Real + 1k Synthetic", "Real + 5k Synthetic"]
        f1s_2 = []
        
        for m in models_2:
            f1s_2.append(data.get(m, {}).get("f1", 0))

        plt.figure(figsize=(10, 6))
        # Gradient Blues
        bars = plt.bar(labels_2, f1s_2, color=['gray', '#3498db', '#2980b9'])
        
        plt.title("Effect of Data Scaling on Performance", fontsize=14, fontweight='bold')
        plt.ylabel("F1 Score (Higher is Better)")
        
        plt.ylim(0.0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Arrow
        if len(f1s_2) >= 3 and f1s_2[2] > f1s_2[1]:
            plt.annotate('Scaling Boost', xy=(2, f1s_2[2]), xytext=(1.5, f1s_2[2]+0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05))

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                         f'{height:.3f}', ha='center', va='bottom', fontsize=12)

        plt.savefig("chart_3_scaling_analysis.png")
        print("Saved: chart_3_scaling_analysis.png")

if __name__ == "__main__":
    plot_results()
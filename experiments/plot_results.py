from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
output_dir = ROOT / "outputs/evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

results = [
    ["K-Means", 26, 52, 0.4931, 20.6368],
    ["Canny + Hough", 0, 0, 0.5376, 0.2875],
    ["HSV", 10, 20, 0.5705, 0.3669],
    ["SegFormer", 2, 4, 0.3788, 0.6018],
    ["Hybrid", 20, 40, 0.4329, 8.3619]
]

methods = [r[0] for r in results]
label_rates = [r[2] for r in results]
ious = [r[3] for r in results]
times = [r[4] for r in results]

csv_path = output_dir / "final_results.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Method",
        "Frames Labeled",
        "Labeling Rate (%)",
        "ROI IoU",
        "Average ALINA Time (s)"
    ])

    writer.writerows(results)

plt.figure(figsize=(10, 5))
plt.bar(methods, label_rates)
plt.ylabel("Frames Reported Labeled (%)")
plt.title("ALINA Labeling Rate by ROI Method")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 60)
plt.tight_layout()
plt.savefig(output_dir / "labeling_rate.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(methods, ious)
plt.ylabel("Intersection over Union")
plt.title("Automatic ROI vs. Manual Reference")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(output_dir / "roi_iou.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(methods, times)
plt.ylabel("Average ALINA Time per Frame (s)")
plt.title("ALINA Processing Runtime by ROI Method")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(output_dir / "runtime.png", dpi=300)
plt.close()

print("Saved evaluation results to:", output_dir)
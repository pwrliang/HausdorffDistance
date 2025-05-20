import nibabel as nib
import monai
import torch
import sys
from datetime import datetime
import json
from monai.metrics import compute_hausdorff_distance

device = torch.device("cuda")

def load_mask(path):
    """Load a .nii /.nii.gz file → Boolean tensor on CPU."""
    im = nib.load(path)
    mask = (im.get_fdata() > 0)          # binary mask in ndarray
    mask_t = torch.as_tensor(mask, dtype=torch.bool)  # → bool tensor
    return mask_t, im.affine                    # keep affine for spacing


file1 = sys.argv[1]
file2 = sys.argv[2]


mask1, affine1 = load_mask(file1)
mask2, affine2 = load_mask(file2)
# add batch & channel dims → [1,1,D,H,W]
mask1 = mask1.unsqueeze(0).unsqueeze(0)
mask2 = mask2.unsqueeze(0).unsqueeze(0)
# mask1 = mask1.to(device).contiguous()
# mask2 = mask2.to(device).contiguous()
import time

repeat_times = {}
total_time = 0
for i in range(5):
    start = time.time()
    hd_AtoB = compute_hausdorff_distance(
        mask1,           # “source” mask
        mask2,             # “target” mask
        percentile=100,           # 100 → exact, 95 → HD95
        directed=True,            # ←–– one‑way!
    )                             # shape [B]  (per item)
    print(f"Directed HD(A→B): {hd_AtoB.item():.4f}")
    end = time.time()
    total_time += end - start
    run_info = {"Algorithm": "MedVoxelHD",
                "Execution": "GPU",
                "ReportedTime": (end - start) * 1000, }
    repeat_times["Repeat" + str(i)] = run_info

repeat_times["AvgTime"] = total_time * 1000 / 5

out = {
    "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Input": {
        "FileA": {"Path": file1},
        "FileB": {"Path": file2},
    },
    "Running": repeat_times
}

with open(sys.argv[3], "w") as f:
    f.write(json.dumps(out, indent=4))

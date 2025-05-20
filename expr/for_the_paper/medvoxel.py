import interface_HD
import torch
from torch.utils.cpp_extension import load
import nibabel as nib
import sys
from datetime import datetime
import json

lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=False)
device = torch.device("cuda")


def nifti_to_pytorch_tensor(nifti_file_path):
    # Load the NIfTI image using nibabel
    img = nib.load(nifti_file_path)

    # Get the image data as a NumPy array.
    # img.get_fdata() is generally recommended as it returns the data
    # as a floating-point type (usually float64) and handles any
    # necessary scaling or type conversions defined in the NIfTI header.
    # The data is typically in (x, y, z) or (x, y, z, t) order for 3D/4D images.
    image_data_numpy = img.get_fdata()
    bool_data = image_data_numpy > 0

    # Convert to PyTorch Boolean tensor
    tensor_bool = torch.from_numpy(bool_data).to(torch.bool)
    # Convert the NumPy array to a PyTorch tensor.
    # torch.from_numpy() creates a tensor that shares memory with the
    # NumPy array. This is efficient, but changes to one will affect
    # the other (if the data types are compatible and the array is writable).
    # If you need a distinct copy, use torch.tensor(image_data_numpy).
    print(f'Tensor shape: {tensor_bool.shape}')
    return tensor_bool


file1 = sys.argv[1]
file2 = sys.argv[2]

img_1 = nifti_to_pytorch_tensor(file1)
img_2 = nifti_to_pytorch_tensor(file2)
img_1 = img_1.to(device).contiguous()
img_2 = img_2.to(device).contiguous()
import time

repeat_times = {}
total_time = 0
for i in range(5):
    start = time.time()
    HD = interface_HD.getHausdorffDistance(img_1, img_2, 1.0)
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

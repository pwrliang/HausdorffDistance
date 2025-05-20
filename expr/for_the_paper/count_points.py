import os
import nibabel as nib
import numpy as np

# --- NIfTI File Processing Logic ---
def get_nifti_non_empty_voxel_count(file_path):
    """
    Loads a NIfTI file and returns the count of non-empty (non-zero) voxels.

    Args:
        file_path (str): The path to the .nii file.

    Returns:
        int or None: The number of non-empty voxels, or None if an error occurs.
    """
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        total_voxels = np.prod(data.shape) # For informational print
        num_non_empty_voxels = np.count_nonzero(data)
        print(f"  - NIfTI Info: Dimensions: {data.shape}, Total Voxels: {total_voxels}, Non-Empty Voxels: {num_non_empty_voxels}")
        return num_non_empty_voxels
    except nib.filebasedimages.ImageFileError as e:
        print(f"  - Error: Could not read NIfTI file '{file_path}'. It might be corrupted or not a valid NIfTI format. Details: {e}")
    except Exception as e:
        print(f"  - An unexpected error occurred while processing NIfTI file '{file_path}': {e}")
    return None

# --- OFF File Processing Logic ---
def get_off_vertex_count(file_path):
    """
    Parses an OFF file and returns the number of vertices.

    Args:
        file_path (str): The path to the .off file.

    Returns:
        int or None: The number of vertices, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

            if not lines:
                print(f"  - Error: OFF file '{file_path}' is empty. Skipping.")
                return None

            header_line_index = 0
            # The first line can be "OFF" (case-insensitive) or comments starting with '#'
            if lines[0].strip().upper() == "OFF":
                header_line_index = 1

            # Skip comment lines
            while header_line_index < len(lines) and lines[header_line_index].strip().startswith('#'):
                header_line_index += 1

            if header_line_index >= len(lines):
                print(f"  - Error: Could not find header line (num_vertices num_faces num_edges) in OFF file '{file_path}'. Skipping.")
                return None

            header_parts = lines[header_line_index].strip().split()

            # We need at least num_vertices and num_faces, though num_edges is often present
            if len(header_parts) < 2:
                print(f"  - Error: Header line in OFF file '{file_path}' is malformed: '{lines[header_line_index].strip()}'. Expected at least 2 numbers. Skipping.")
                return None

            try:
                num_vertices = int(header_parts[0])
                print(f"  - OFF Info: Vertices: {num_vertices}")
                return num_vertices
            except ValueError:
                print(f"  - Error: Could not parse vertex count from header in OFF file '{file_path}': '{lines[header_line_index].strip()}'. Skipping.")
    except FileNotFoundError: # Should be rare with os.walk, but good practice
        print(f"  - Error: OFF file not found at '{file_path}'. Skipping.")
    except Exception as e:
        print(f"  - An unexpected error occurred while processing OFF file '{file_path}': {e}")
    return None

# --- Main Processing Function ---
def process_folder_recursively(folder_path):
    """
    Recursively scans a folder for .nii and .off files, and calculates min/max/median/average
    non-empty voxels for .nii and min/max/median/average vertices for .off files.
    """
    min_nii_voxels = float('inf')
    max_nii_voxels = float('-inf')
    nii_voxel_counts_list = [] # To store all voxel counts for median and average calculation
    nii_files_found = 0
    nii_files_processed = 0

    min_off_vertices = float('inf')
    max_off_vertices = float('-inf')
    off_vertex_counts_list = [] # To store all vertex counts for median and average calculation
    off_files_found = 0
    off_files_processed = 0

    if not os.path.isdir(folder_path):
        print(f"Error: Root folder not found at '{folder_path}'")
        return

    print(f"Recursively scanning folder: {folder_path}\n")

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            if filename.lower().endswith(".nii"):
                print(f"Processing NIfTI file: {file_path}")
                nii_files_found += 1
                voxel_count = get_nifti_non_empty_voxel_count(file_path)
                if voxel_count is not None:
                    min_nii_voxels = min(min_nii_voxels, voxel_count)
                    max_nii_voxels = max(max_nii_voxels, voxel_count)
                    nii_voxel_counts_list.append(voxel_count)
                    nii_files_processed += 1

            elif filename.lower().endswith(".nii.gz"):
                print(f"Skipping compressed NIfTI file: {file_path}. Please decompress to '.nii' for processing.")

            elif filename.lower().endswith(".off"):
                print(f"Processing OFF file: {file_path}")
                off_files_found += 1
                vertex_count = get_off_vertex_count(file_path)
                if vertex_count is not None:
                    min_off_vertices = min(min_off_vertices, vertex_count)
                    max_off_vertices = max(max_off_vertices, vertex_count)
                    off_vertex_counts_list.append(vertex_count)
                    off_files_processed += 1

    print("\n--- Processing Summary ---")

    # NIfTI Summary
    if nii_files_found == 0:
        print("No .nii files found.")
    elif nii_files_processed == 0:
        print(f"{nii_files_found} .nii files found, but none could be successfully processed.")
    else:
        median_nii_voxels = np.median(nii_voxel_counts_list)
        average_nii_voxels = np.mean(nii_voxel_counts_list)
        print(f"NIfTI (.nii) Files Processed: {nii_files_processed} (out of {nii_files_found} found)")
        print(f"  Min Non-Empty Voxels: {int(min_nii_voxels)}")
        print(f"  Max Non-Empty Voxels: {int(max_nii_voxels)}")
        print(f"  Median Non-Empty Voxels: {median_nii_voxels:.2f}")
        print(f"  Average Non-Empty Voxels: {average_nii_voxels:.2f}")


    print("-" * 30) # Separator

    # OFF Summary
    if off_files_found == 0:
        print("No .off files found.")
    elif off_files_processed == 0:
        print(f"{off_files_found} .off files found, but none could be successfully processed.")
    else:
        median_off_vertices = np.median(off_vertex_counts_list)
        average_off_vertices = np.mean(off_vertex_counts_list)
        print(f"OFF (.off) Files Processed: {off_files_processed} (out of {off_files_found} found)")
        print(f"  Min Vertices: {int(min_off_vertices)}")
        print(f"  Max Vertices: {int(max_off_vertices)}")
        print(f"  Median Vertices: {median_off_vertices:.2f}")
        print(f"  Average Vertices: {average_off_vertices:.2f}")

if __name__ == "__main__":
    folder_to_scan = input("Enter the path to the folder to recursively scan for .nii and .off files: ")
    process_folder_recursively(folder_to_scan)

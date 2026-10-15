import os
import csv
import h5py
import numpy as np
import torch

# base_folder = "test_folder"
base_folder = "/lustre/work/pdl18409/shared/data/optimisation/raw/var/lib/docker/volumes/shared-data/_data/claraserv3/2026/3/30"
# output_file = "image_outputs"
output_file = "/lustre/work/pdl18409/udld6225f/workspace/image_outputs"
# files_for_safe = 2
files_for_safe = 1000

# Set to "cuda:0", "cuda:1", etc. to select a GPU.
# Use "cpu" to force CPU processing.
# DEVICE = "cuda:0"
DEVICE = "cpu"

# Camera conversions:
camera_conversions = {"cla_s02_dia_scr_02": [1.2200e-05, 1.2200e-05],
                      "cla_s02_dia_scr_03": [1.4380e-05, 1.4380e-05],
                      "cla_s03_dia_scr_01": [2.4469e-05, 2.4469e-05],
                      "cla_s04_dia_scr_01": [2.3993e-05, 2.3993e-05],
                      "cla_s04_dia_scr_02": [2.3540e-05, 2.3540e-05],
                      "cla_s04_dia_scr_03": [1.2650e-05, 1.2650e-05],
                      "cla_s04_dia_scr_04": [2.2349e-05, 2.2349e-05],
                      "cla_s04_dia_scr_05": [2.3501e-05, 2.3501e-05],
                      "cla_s05_dia_scr_01": [1.2780e-05, 1.2780e-05],
                      "cla_vbc_dia_scr_01": [6.5000e-06, 6.5000e-06],
                      "cla_s06_dia_scr_01": [2.3371e-05, 2.3371e-05],
                      "cla_s06_dia_scr_02": [1.2143e-05, 1.2143e-05],
                      "cla_s07_dia_scr_01": [2.3196e-05, 2.3196e-05],
                      "cla_s07_dia_scr_02": [1.2614e-05, 1.2614e-05],
                      "cla_s07_dcp_scr_01": [0.0010, 0.0010],
                      "cla_s07_dia_scr_03": [2.3938e-05, 2.3938e-05],
                      "cla_s07_dia_scr_04": [1.2644e-05, 1.2644e-05],
                      "cla_fea_dia_scr_01": [2.2437e-05, 2.2437e-05],
                      "cla_fea_dia_scr_02": [2.3316e-05, 2.3316e-05],
                      "cla_fea_dia_col_03_scr": [0.0010, 0.0010],
                      "cla_fea_dia_scr_03": [2.4374e-05, 2.4374e-05],
                      "cla_fea_dia_scr_04": [2.3331e-05, 2.3331e-05],
                      "cla_feh_dia_scr_01": [1.1641e-05, 1.1641e-05],
                      "cla_feh_dia_scr_02": [1.1784e-05, 1.1784e-05],
                      "cla_feh_dia_scr_03": [2.1198e-05, 2.1198e-05],
                      "cla_feh_dia_scr_04": [2.0951e-05, 2.0951e-05],
                      "cla_fed_dia_scr_01": [1.0634e-05, 1.0634e-05],
                      "cla_fed_dia_scr_02": [4.5345e-05, 4.5345e-05]}

@torch.inference_mode()
def image_center_and_deviation(img, width_mm, height_mm):
    """
    Calculate weighted center and standard deviation using CuPy.
    The returned values are Python floats on the CPU.
    """
    weights = img.to(torch.float64)
    total = weights.sum().item()

    if not np.isfinite(total) or total == 0:
        return (np.nan, np.nan), (np.nan, np.nan)

    # h, w = img.shape[:2]

    # Convert pixel coordinates to real-world coordinates
    # x = (np.arange(w) - 0.5 * w) * width_mm / w
    # y = (np.arange(h) - 0.5 * h) * height_mm / h

    # Weighted center
    # cx = np.sum(weights * x[None, :]) / total
    # cy = np.sum(weights * y[:, None]) / total

    # Weighted standard deviation
    # sx = np.sqrt(np.sum(weights * (x[None, :] - cx)**2) / total)
    # sy = np.sqrt(np.sum(weights * (y[:, None] - cy)**2) / total)

    # return (cx, cy), (sx, sy)

    # New version
    h, w = map(int, img.shape)

    width_mm = float(width_mm)
    height_mm = float(height_mm)

    # Coordinates centered on the image, preserving the original
    # coordinate convention.
    x = (
                torch.arange(w, dtype=torch.float64, device=img.device)
                - 0.5 * w
        ) * (width_mm / w)

    y = (
                torch.arange(h, dtype=torch.float64, device=img.device)
                - 0.5 * h
        ) * (height_mm / h)

    # Sum intensity by column and row.
    # This avoids allocating full-sized coordinate grids.
    col_sum = weights.sum(dim=0)
    row_sum = weights.sum(dim=1)

    # Intensity-weighted center.
    cx = (col_sum * x).sum() / total
    cy = (row_sum * y).sum() / total

    # Weighted population standard deviations.
    sx = torch.sqrt(
        (col_sum * (x - cx).square()).sum() / total
    )
    sy = torch.sqrt(
        (row_sum * (y - cy).square()).sum() / total
    )

    # Transfer only four scalar results to the CPU.
    cx, cy, sx, sy = torch.stack((cx, cy, sx, sy)).cpu().tolist()

    return (cx, cy), (sx, sy)

@torch.inference_mode()
def treat_file(dataset, camera_conversion):
    device = torch.device(DEVICE)

    # Upload the image to the GPU.
    img = torch.as_tensor(dataset, device=device).to(torch.int32)

    # Convert images (filter)
    # dataset_converted = np.maximum(dataset.astype(np.int32) - dataset.max() // 10, 0).astype(np.uint16)
    threshold = img.max().item() // 10
    cleaned = torch.clamp(
        img - threshold,
        min=0,
    )

    # Calculate center and deviation
    real_width = dataset.shape[0] * camera_conversion[0]
    real_height = dataset.shape[1] * camera_conversion[1]
    return image_center_and_deviation(cleaned, real_width, real_height)

def write_csv(content, index=0):
    if index:
        file_name = f"{output_file}-{index:05d}.csv"
    else:
        file_name = f"{output_file}.csv"

    # Write output
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["File name", "Capture number", "cx", "cy", "sx", "sy"])
        writer.writerows(content)

def main():
    # Collect images
    csv_contents = []
    file_index = 1
    file_count = 0

    for current_dir, subdirs, files in os.walk(base_folder):
        for file in files:
            # file_found = False
            # Check existence of camera
            for camera_conversion in camera_conversions:
                if file.lower().replace("cam", "scr").replace("-", "_").startswith(camera_conversion):
                    print(file)
                    file_count += 1
                    if file_count > files_for_safe:
                        write_csv(csv_contents, index=file_index)
                        file_index += 1
                        file_count = 0
                    # file_found = True
                    # Extract data
                    try:
                        with h5py.File(os.path.join(current_dir, file), "r") as f:
                            for key in f.keys():
                                centers, deviation = treat_file(f[key][:], camera_conversions[camera_conversion])
                                csv_contents.append([file, key, centers[0], centers[1], deviation[0], deviation[1]])
                    except Exception as e:
                        print("File did not work")
                        print(e)
            # if not file_found:
            # print(f"File {file_name} not found")

    write_csv(csv_contents)

if __name__ == "__main__":
    main()

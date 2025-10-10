import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import math
import kagglehub

# Download latest version
path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")

# Path ke folder dataset
INFECTED_DIR = os.path.join(path, "cell_images", "Parasitized")
HEALTHY_DIR = os.path.join(path, "cell_images", "Uninfected")

# Folder output
INFECTED_OUT = "parasitized_processed"
HEALTHY_OUT = "uninfected_processed"

# Buat folder output jika belum ada
os.makedirs(INFECTED_OUT, exist_ok=True)
os.makedirs(HEALTHY_OUT, exist_ok=True)

# Parameter LBP
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"

# Parameter Gabor
GABOR_KSIZE = 31
GABOR_SIGMA = 4.0
GABOR_GAMMA = 0.5
GABOR_PSI = 0
GABOR_LAMBDAS = [8.0]   # bisa ditambah [4.0, 8.0, 16.0]
GABOR_THETAS = [0, math.pi/4, math.pi/2, 3*math.pi/4]  # 4 orientasi

def build_gabor_kernels():
    kernels = []
    for lam in GABOR_LAMBDAS:
        for theta in GABOR_THETAS:
            kern = cv2.getGaborKernel((GABOR_KSIZE, GABOR_KSIZE), 
                                      GABOR_SIGMA, theta, lam, 
                                      GABOR_GAMMA, GABOR_PSI, 
                                      ktype=cv2.CV_32F)
            kernels.append(kern)
    return kernels

def apply_lbp_then_gabor(img_gray, kernels):
    """ Terapkan LBP dulu, lalu Gabor filter bank """
    # 1. LBP
    lbp = local_binary_pattern(img_gray, LBP_P, LBP_R, method=LBP_METHOD)
    lbp_img = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))

    # 2. Gabor di atas hasil LBP
    responses = []
    for kern in kernels:
        resp = cv2.filter2D(lbp_img, cv2.CV_8UC3, kern)
        responses.append(resp.astype(np.float32))

    gabor_combined = np.mean(responses, axis=0).astype(np.uint8)
    return gabor_combined

def process_folder(input_dir, output_dir, kernels):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for fname in tqdm(files, desc=f"Processing {input_dir}"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        processed = apply_lbp_then_gabor(img, kernels)
        cv2.imwrite(out_path, processed)

def main():
    kernels = build_gabor_kernels()
    process_folder(INFECTED_DIR, INFECTED_OUT, kernels)
    process_folder(HEALTHY_DIR, HEALTHY_OUT, kernels)
    print("Proses selesai ✅")

if __name__ == "__main__":
    main()
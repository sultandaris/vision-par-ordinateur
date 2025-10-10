import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import math

sehat_path = "/Users/endah/Desktop/kuliah/sehat_sampel"
parasit_path = "/Users/endah/Desktop/kuliah/parasit_sampel"

# Parameter LBP
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"

# Parameter Gabor
GABOR_KSIZE = 31
GABOR_SIGMA = 4.0
GABOR_GAMMA = 0.5
GABOR_PSI = 0
GABOR_LAMBDAS = [8.0]
GABOR_THETAS = [0, math.pi/4, math.pi/2, 3*math.pi/4]

# Folder output
INFECTED_OUT = "parasit_output_sampel"
HEALTHY_OUT = "sehat_output_sampel"

# Buat folder output jika belum ada
os.makedirs(INFECTED_OUT, exist_ok=True)
os.makedirs(HEALTHY_OUT, exist_ok=True)

all_list = os.listdir(parasit_path) + os.listdir(sehat_path)

def L_feature_extraction(image, P=8, R=1):
    """Apply Local Binary Pattern (LBP) filter to the image"""
    lbp = local_binary_pattern(image, P, R, method="uniform")
    # Normalisasi sama seperti file_kodingan_komplit
    lbp_norm = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min()))
    return lbp_norm

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

def apply_lbp_then_gabor(image):
    """Terapkan LBP kemudian Gabor pada gambar - sama seperti file_kodingan_komplit"""
    # Step 1: Terapkan LBP
    lbp_result = L_feature_extraction(image, LBP_P, LBP_R)
    
    # Step 2: Build Gabor kernels
    kernels = build_gabor_kernels()
    
    # Step 3: Terapkan multiple Gabor pada hasil LBP
    responses = []
    for kern in kernels:
        resp = cv2.filter2D(lbp_result, cv2.CV_8UC3, kern)
        responses.append(resp.astype(np.float32))
    
    # Step 4: Rata-rata dari semua orientasi
    gabor_combined = np.mean(responses, axis=0).astype(np.uint8)
    
    return lbp_result, gabor_combined

def create_comparison(original, lbp_img, gabor_img, title):
    """Membuat gambar perbandingan side-by-side"""
    h, w = original.shape
    
    # Pastikan semua gambar memiliki ukuran yang sama
    lbp_resized = cv2.resize(lbp_img, (w, h))
    gabor_resized = cv2.resize(gabor_img, (w, h))
    
    # Gabungkan gambar secara horizontal
    comparison = np.hstack([original, lbp_resized, gabor_resized])
    
    # Tambahkan label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1
    
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison, "LBP", (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison, "LBP+Gabor", (2*w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison, title, (10, h - 10), font, font_scale, color, thickness)
    
    return comparison

def analyze_statistics(lbp_values, gabor_values, image_type):
    """Analisis statistik untuk gambar LBP dan Gabor"""
    stats = {
        'type': image_type,
        'lbp_mean': np.mean(lbp_values),
        'lbp_std': np.std(lbp_values),
        'lbp_min': np.min(lbp_values),
        'lbp_max': np.max(lbp_values),
        'gabor_mean': np.mean(gabor_values),
        'gabor_std': np.std(gabor_values),
        'gabor_min': np.min(gabor_values),
        'gabor_max': np.max(gabor_values)
    }
    return stats

# Inisialisasi list untuk menyimpan statistik
parasit_stats = []
sehat_stats = []

print("Memproses gambar dengan LBP kemudian Gabor...")

for item in tqdm(all_list, desc="Processing images"):
    if item in os.listdir(parasit_path):
        img_path = os.path.join(parasit_path, item)
        output_dir = INFECTED_OUT
        image_type = "PARASIT"
    else:
        img_path = os.path.join(sehat_path, item)
        output_dir = HEALTHY_OUT
        image_type = "SEHAT"

    # Baca gambar
    image = cv2.imread(img_path)
    if image is None:
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Terapkan LBP kemudian Gabor
    lbp_result, gabor_result = apply_lbp_then_gabor(gray)

    # Buat subfolder untuk menyimpan hasil
    lbp_dir = os.path.join(output_dir, "LBP")
    gabor_dir = os.path.join(output_dir, "LBP_Gabor")
    comparison_dir = os.path.join(output_dir, "Comparison")
    
    os.makedirs(lbp_dir, exist_ok=True)
    os.makedirs(gabor_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # Simpan hasil LBP dan LBP+Gabor
    cv2.imwrite(os.path.join(lbp_dir, item), lbp_result)
    cv2.imwrite(os.path.join(gabor_dir, item), gabor_result)
    
    # Buat dan simpan gambar perbandingan
    comparison = create_comparison(gray, lbp_result, gabor_result, f"{image_type}: {item}")
    cv2.imwrite(os.path.join(comparison_dir, f"comparison_{item}"), comparison)
    
    # Kumpulkan statistik
    stats = analyze_statistics(lbp_result, gabor_result, image_type)
    if image_type == "PARASIT":
        parasit_stats.append(stats)
    else:
        sehat_stats.append(stats)

print("Proses selesai! Hasil tersimpan di folder output.")

# Tampilkan statistik perbandingan
print("\n" + "="*60)
print("ANALISIS STATISTIK PERBANDINGAN")
print("="*60)

if parasit_stats:
    print(f"\nGAMBAR PARASIT ({len(parasit_stats)} gambar):")
    avg_lbp_mean = np.mean([s['lbp_mean'] for s in parasit_stats])
    avg_lbp_std = np.mean([s['lbp_std'] for s in parasit_stats])
    avg_gabor_mean = np.mean([s['gabor_mean'] for s in parasit_stats])
    avg_gabor_std = np.mean([s['gabor_std'] for s in parasit_stats])
    
    print(f"LBP - Mean: {avg_lbp_mean:.4f}, Std: {avg_lbp_std:.4f}")
    print(f"LBP+Gabor - Mean: {avg_gabor_mean:.4f}, Std: {avg_gabor_std:.4f}")

if sehat_stats:
    print(f"\nGAMBAR SEHAT ({len(sehat_stats)} gambar):")
    avg_lbp_mean = np.mean([s['lbp_mean'] for s in sehat_stats])
    avg_lbp_std = np.mean([s['lbp_std'] for s in sehat_stats])
    avg_gabor_mean = np.mean([s['gabor_mean'] for s in sehat_stats])
    avg_gabor_std = np.mean([s['gabor_std'] for s in sehat_stats])
    
    print(f"LBP - Mean: {avg_lbp_mean:.4f}, Std: {avg_lbp_std:.4f}")
    print(f"LBP+Gabor - Mean: {avg_gabor_mean:.4f}, Std: {avg_gabor_std:.4f}")

if parasit_stats and sehat_stats:
    print(f"\nPERBANDINGAN:")
    parasit_lbp_mean = np.mean([s['lbp_mean'] for s in parasit_stats])
    sehat_lbp_mean = np.mean([s['lbp_mean'] for s in sehat_stats])
    parasit_gabor_mean = np.mean([s['gabor_mean'] for s in parasit_stats])
    sehat_gabor_mean = np.mean([s['gabor_mean'] for s in sehat_stats])
    
    lbp_diff = abs(parasit_lbp_mean - sehat_lbp_mean)
    gabor_diff = abs(parasit_gabor_mean - sehat_gabor_mean)
    
    print(f"Selisih rata-rata LBP: {lbp_diff:.4f}")
    print(f"Selisih rata-rata LBP+Gabor: {gabor_diff:.4f}")
    
    if gabor_diff > lbp_diff:
        print("Kombinasi LBP+Gabor menunjukkan perbedaan yang lebih besar antara gambar sehat dan parasit")
    else:
        print("LBP saja sudah cukup untuk membedakan gambar sehat dan parasit")

print("\n" + "="*60)# ...existing code...

def create_comparison(original, gabor_img, title):
    """Membuat gambar perbandingan side-by-side: Original vs LBP+Gabor"""
    h, w = original.shape
    
    # Pastikan gambar Gabor memiliki ukuran yang sama
    gabor_resized = cv2.resize(gabor_img, (w, h))
    
    # Gabungkan gambar secara horizontal (hanya original dan gabor)
    comparison = np.hstack([original, gabor_resized])
    
    # Tambahkan label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1
    
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison, "LBP+Gabor", (w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison, title, (10, h - 10), font, font_scale, color, thickness)
    
    return comparison

# ...existing code...

for item in tqdm(all_list, desc="Processing images"):
    if item in os.listdir(parasit_path):
        img_path = os.path.join(parasit_path, item)
        output_dir = INFECTED_OUT
        image_type = "PARASIT"
    else:
        img_path = os.path.join(sehat_path, item)
        output_dir = HEALTHY_OUT
        image_type = "SEHAT"

    # Baca gambar
    image = cv2.imread(img_path)
    if image is None:
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Terapkan LBP kemudian Gabor
    lbp_result, gabor_result = apply_lbp_then_gabor(gray)

    # Buat subfolder untuk menyimpan hasil
    gabor_dir = os.path.join(output_dir, "LBP_Gabor")
    comparison_dir = os.path.join(output_dir, "Comparison")
    
    os.makedirs(gabor_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # Simpan hasil LBP dan LBP+Gabor
    cv2.imwrite(os.path.join(gabor_dir, item), gabor_result)
    
    # Buat dan simpan gambar perbandingan (hanya original vs LBP+Gabor)
    comparison = create_comparison(gray, gabor_result, f"{image_type}: {item}")
    cv2.imwrite(os.path.join(comparison_dir, f"comparison_{item}"), comparison)
    
    # Kumpulkan statistik
    stats = analyze_statistics(lbp_result, gabor_result, image_type)
    if image_type == "PARASIT":
        parasit_stats.append(stats)
    else:
        sehat_stats.append(stats)


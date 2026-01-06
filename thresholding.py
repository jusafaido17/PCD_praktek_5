import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk 
import os # Import os untuk memeriksa keberadaan file

# --- KONFIGURASI FILE INPUT ---
BABY_FILENAME = 'monster02.jpg'    
CLOUD_FILENAME = '42.JPG'
OUTPUT_BW_FILENAME = 'bw4.jpg' 

def thresholding_compositing_cropping_praktek_5(baby_file, cloud_file, output_bw_file):
    
    # 1. MEMBACA DAN KONVERSI CITRA KE RGB
    baby_bgr = cv2.imread(baby_file)
    cloud_bgr = cv2.imread(cloud_file)
    
    if baby_bgr is None or cloud_bgr is None:
        missing_file = baby_file if baby_bgr is None else cloud_file
        print(f"❌ ERROR: Tidak dapat memuat file '{missing_file}'. Pastikan file ada di folder yang sama.")
        return

    # Konversi semua citra ke RGB di awal
    baby_rgb_working = cv2.cvtColor(baby_bgr, cv2.COLOR_BGR2RGB)
    cloud_rgb_working = cv2.cvtColor(cloud_bgr, cv2.COLOR_BGR2RGB)
    
    # Konversi citra objek ke grayscale untuk Otsu
    baby_gray = cv2.cvtColor(baby_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. THRESHOLDING OTSu DAN PENYEMPURNAAN MASKER (Meniru MATLAB Lines 5-11)
    
    # Thresholding Otsu
    ret, baby_bw_otsu = cv2.threshold(baby_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Konversi biner ke boolean (True/False)
    baby_mask_bool = baby_bw_otsu > 0 
    
    # Imfill(..., 'holes')
    baby_mask_filled = binary_fill_holes(baby_mask_bool)
    
    # Erosi (imerode)
    kernel = np.ones((3,3), np.uint8) 
    baby_bw_final = cv2.erode((baby_mask_filled * 255).astype(np.uint8), kernel, iterations=1)
    
    # Mask boolean final
    final_mask_bool = baby_bw_final > 0
    
    cv2.imwrite(output_bw_file, baby_bw_final)
    print(f"✅ Citra biner hasil penyempurnaan disimpan sebagai: {output_bw_file}")
    
    # 3. IMAGE COMPOSITING (PENGGABUNGAN)

    h, w, _ = baby_rgb_working.shape
    cloud_resized = cv2.resize(cloud_rgb_working, (w, h)) 
    
    mask_3d = np.dstack((final_mask_bool, final_mask_bool, final_mask_bool))
    
    # Hitung Latar Belakang
    background = cloud_resized.copy()
    background[mask_3d] = 0 
    
    # Hitung Foreground
    foreground = baby_rgb_working.copy()
    foreground[~mask_3d] = 0 
    
    # Compositing: Penjumlahan
    rgb_composited = background + foreground 
    
    
    # 4. CROPPING CITRA DAN BATAS (Meniru MATLAB Lines 30-52)
    
    # Buat baby_RGB (objek berwarna dengan background hitam)
    baby_rgb_black_bg = baby_rgb_working.copy()
    inverse_mask_3d = ~mask_3d
    baby_rgb_black_bg[inverse_mask_3d] = 0 
    
    # Temukan koordinat
    row_indices, col_indices = np.where(final_mask_bool)
    
    if len(row_indices) == 0:
        print("Peringatan: Mask biner kosong, Cropping dilewati.")
        return

    # Tentukan batas bounding box
    min_row, max_row = np.min(row_indices), np.max(row_indices)
    min_col, max_col = np.min(col_indices), np.max(col_indices)
    
    # Cropping RGB 
    RGB_cropped_black_bg = baby_rgb_black_bg[min_row:max_row+1, min_col:max_col+1] 
    
    # Cropping Biner 
    bw_cropped = baby_bw_final[min_row:max_row+1, min_col:max_col+1]
    
    # Temukan batas (bwboundaries)
    contours, hierarchy = cv2.findContours(bw_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # 5. VISUALISASI MATPLOTLIB
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flat
    plt.suptitle("Praktek 5: Thresholding, Compositing, Cropping, dan Boundary", fontsize=16)

    # Plot 1: Citra Asli 
    axes_flat[0].imshow(baby_rgb_working)
    axes_flat[0].set_title(f'1. Citra Asli', fontweight='bold')
    axes_flat[0].axis('off')

    # Plot 2: Citra Biner Final
    axes_flat[1].imshow(baby_bw_final, cmap='gray')
    axes_flat[1].set_title('2. Biner Hasil Penyempurnaan', fontweight='bold')
    axes_flat[1].axis('off')

    # Plot 3: Citra Background 
    axes_flat[2].imshow(cloud_rgb_working)
    axes_flat[2].set_title('3. Citra Background (Cloud)', fontweight='bold')
    axes_flat[2].axis('off')

    # Plot 4: Citra Compositing
    axes_flat[3].imshow(rgb_composited)
    axes_flat[3].set_title('4. Composited Image (Cloud + Foreground)', fontweight='bold')
    axes_flat[3].axis('off')
    
    # Plot 5: Hasil Cropping RGB + Batas
    axes_flat[4].imshow(RGB_cropped_black_bg)
    axes_flat[4].set_title('5. Cropping RGB + Batas', fontweight='bold')
    axes_flat[4].axis('off')

    # Tambahkan Batas (Boundary) pada Plot 5
    for contour in contours:
        axes_flat[4].plot(contour[:, 0, 0], contour[:, 0, 1], 'g', linewidth=2)
    
    # Plot 6: Cropping Biner
    axes_flat[5].imshow(bw_cropped, cmap='gray')
    axes_flat[5].set_title('6. Cropping Biner', fontweight='bold')
    axes_flat[5].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.3)
    plt.show()

# --- JALANKAN PROGRAM ---
thresholding_compositing_cropping_praktek_5(BABY_FILENAME, CLOUD_FILENAME, OUTPUT_BW_FILENAME)
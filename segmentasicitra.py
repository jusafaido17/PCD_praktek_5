import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import chan_vese # Implementasi region-based Active Contour
import os

# --- KONFIGURASI FILE INPUT ---
FILENAME = 'sydney.jpg' # Diperbarui sesuai traceback Anda    
MAX_ITERATIONS = 350 # Sesuai dengan parameter di MATLAB

def active_contour_segmentation(filename, max_iter):
    
    # 1. MEMBACA CITRA
    I_bgr = cv2.imread(filename)
    
    if I_bgr is None:
        print(f"❌ ERROR: Tidak dapat memuat file '{filename}'. Pastikan file ada.")
        return

    # Konversi BGR ke RGB untuk Matplotlib
    I_rgb = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. KONVERSI KE GRAYSCALE
    J_gray = cv2.cvtColor(I_rgb, cv2.COLOR_RGB2GRAY)
    
    # 3. MEMBUAT INISIAL MASKING
    h, w = J_gray.shape
    m = np.zeros(J_gray.shape, dtype=np.uint8)
    
    # m(111:231,123:243) = 1; (disesuaikan dengan 0-indexing Python)
    m[200:400, 400:600] = 1 
    
    # 4. SEGMENTASI ACTIVE CONTOUR
    J_norm = J_gray / 255.0
    
    # ❗ PERBAIKAN ERROR: Mengganti max_iter -> max_num_iter dan init_level -> init_level_set ❗
    seg_bool = chan_vese(J_norm, m, max_num_iter=max_iter, dt=0.5, init_level_set=m, extended_output=False)
    
    # Konversi hasil biner (boolean) ke 0-255 untuk visualisasi
    seg_biner = seg_bool.astype(np.uint8) * 255
    
    # 5. MENAMPILKAN HASIL
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Praktek 5: Segmentasi Active Contour (Chan-Vese)", fontsize=16)
    axes_flat = axes.flat

    # Plot 1: Citra rgb asli
    axes_flat[0].imshow(I_rgb)
    axes_flat[0].set_title('1. Citra RGB Asli')
    axes_flat[0].axis('off')

    # Plot 2: Inisial masking
    axes_flat[1].imshow(m, cmap='gray')
    axes_flat[1].set_title('2. Inisial Masking')
    axes_flat[1].axis('off')

    # Plot 3: Citra biner hasil segmentasi
    axes_flat[2].imshow(seg_biner, cmap='gray')
    axes_flat[2].set_title('3. Citra Biner Hasil Segmentasi')
    axes_flat[2].axis('off')

    # Plot 4: Citra rgb hasil segmentasi (Menampilkan overlay batas hasil di atas RGB asli)
    axes_flat[3].imshow(I_rgb)
    axes_flat[3].contour(seg_bool, [0.5], colors='r', linewidths=2) 
    axes_flat[3].set_title('4. Citra RGB Hasil Segmentasi (Batas Merah)')
    axes_flat[3].axis('off')

    plt.subplots_adjust(hspace=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True) 

# --- JALANKAN PROGRAM ---
print(f"Memulai segmentasi Active Contour pada '{FILENAME}' dengan {MAX_ITERATIONS} iterasi.")
active_contour_segmentation(FILENAME, MAX_ITERATIONS)
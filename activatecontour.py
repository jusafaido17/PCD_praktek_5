import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import chan_vese
from skimage import img_as_float

# --- KONFIGURASI ---
INPUT_FILENAME = 'cars_14.jpg' 
ITERATIONS = 350 

# Koordinat untuk Inisial Masking
ROW_START, ROW_END = 110, 231
COL_START, COL_END = 122, 243

def active_contour_segmentation(filename, iterations):
    
    I = cv2.imread(filename)
    if I is None:
        print(f"❌ ERROR: Tidak dapat memuat gambar: {filename}")
        print("Pastikan file gambar ada di folder yang sama (PWT.jpg).")
        return
    
    I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J_float = img_as_float(J) 
    
    # 3. Membuat Inisial Masking (m)
    m = np.zeros_like(J, dtype=bool) 
    m[ROW_START:ROW_END, COL_START:COL_END] = True 
    
    # 4. Segmentasi Citra menggunakan Active Contour (seg)
    # ❗ SOLUSI AKHIR: Menggunakan Argumen Posisi untuk mengatasi masalah versi library.
    seg = chan_vese(J_float, 
                    0.25,       # Posisi ke-2: mu (Default value)
                    1.0,        # Posisi ke-3: lambda1 (Default value)
                    1.0,        # Posisi ke-4: lambda2 (Default value)
                    1e-3,       # Posisi ke-5: tol (Default value)
                    iterations, # Posisi ke-6: Iterasi (yang selalu gagal jika menggunakan keyword)
                    0.5,        # Posisi ke-7: dt (Default value)
                    init_level_set=m,
                    extended_output=False)
    
    # Lanjutkan dengan visualisasi
    seg_binary = seg.astype(np.uint8) * 255
    seg_mask = seg_binary > 0 
    I_overlay_final = I_rgb * seg_mask[:, :, np.newaxis]
    
    
    # --- VISUALISASI MATPLOTLIB ---
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.suptitle(f"Segmentasi Citra Menggunakan Active Contour", fontsize=16)

    # Plot 1: Citra RGB Asli
    axes[0, 0].imshow(I_rgb)
    axes[0, 0].set_title('1. Citra RGB Asli', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Plot 2: Inisial Masking
    axes[0, 1].imshow(m, cmap='gray')
    axes[0, 1].set_title('2. Inisial Masking', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Plot 3: Citra Biner Hasil Segmentasi
    axes[1, 0].imshow(seg_binary, cmap='gray')
    axes[1, 0].set_title('3. Citra Biner Hasil Segmentasi', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Plot 4: Citra RGB Hasil Segmentasi (Overlay)
    axes[1, 1].imshow(I_overlay_final)
    axes[1, 1].set_title('4. Citra RGB Hasil Segmentasi (Objek)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Menghindari tabrakan judul
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.25) 
    plt.show()

# --- JALANKAN PROGRAM ---
active_contour_segmentation(INPUT_FILENAME, ITERATIONS)
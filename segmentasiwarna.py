import cv2
import numpy as np
import matplotlib.pyplot as plt
# Menggunakan skimage karena konversi HSV-nya secara default menghasilkan nilai Hue 0.0-1.0
from skimage.color import rgb2hsv
from skimage import img_as_float

# --- KONFIGURASI ---
# GANTI 'color_image.jpg' dengan nama file gambar Anda yang memiliki warna hijau!
INPUT_FILENAME = 'cars_04.jpg' 

# Threshold Hue untuk warna Hijau, sesuai dengan dokumen (0.26 sampai 0.36)
HUE_MIN = 0.26
HUE_MAX = 0.36

def color_segmentation_hsv(filename, h_min, h_max):
    
    # 1. Membaca Citra (OpenCV membaca dalam BGR)
    I_bgr = cv2.imread(filename)
    if I_bgr is None:
        print(f"❌ ERROR: Tidak dapat memuat gambar: {filename}")
        print("Pastikan file gambar ada di folder yang sama dan memiliki objek hijau.")
        return

    # Konversi BGR ke RGB (karena matplotlib dan skimage menggunakan RGB)
    I_rgb = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB)
    
    # Konversi citra ke float (diperlukan oleh skimage)
    I_float = img_as_float(I_rgb)
    
    # 2. Konversi Ruang Warna RGB ke HSV
    # Hasil konversi ini memiliki Hue (H) dalam rentang [0.0, 1.0]
    I_hsv = rgb2hsv(I_float)
    
    # 3. Ekstraksi Komponen Hue (H_aksen)
    # Hue berada pada channel pertama (indeks 0)
    H_aksen = I_hsv[:, :, 0]
    
    # 4. Segmentasi Warna (Thresholding)
    # Terapkan thresholding pada komponen Hue: (H_aksen > 0.26) & (H_aksen < 0.36)
    mask_h = (H_aksen > h_min) & (H_aksen < h_max)
    
    # Masking gabungan S dan V (untuk hasil yang lebih baik, memastikan warna cerah)
    # S > 0.3 dan V > 0.3 (opsional, untuk menyingkirkan bayangan/warna kusam)
    mask_s = I_hsv[:, :, 1] > 0.3 
    mask_v = I_hsv[:, :, 2] > 0.3
    
    # Mask final hanya mengambil area yang memenuhi kriteria Hue, Saturation, dan Value
    final_mask = mask_h & mask_s & mask_v

    # 5. Menerapkan Mask pada Citra Asli
    # Membuat citra hanya berisi objek yang tersegmentasi
    segmented_object = I_rgb * final_mask[:, :, np.newaxis]
    
    # Membuat citra biner (mask) untuk ditampilkan
    binary_mask = final_mask.astype(np.uint8) * 255
    
    # --- VISUALISASI MATPLOTLIB ---
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle(f"Praktek 5: Segmentasi Warna Hijau", fontsize=16)

    # Plot 1: Citra RGB Asli
    axes[0].imshow(I_rgb)
    axes[0].set_title('1. Citra RGB Asli', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot 2: Mask Biner Hasil Segmentasi
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title('2. Mask Biner (H: 0.26-0.36)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Plot 3: Objek Hasil Segmentasi (Overlay)
    axes[2].imshow(segmented_object)
    axes[2].set_title('3. Objek Hijau yang Tersegmentasi', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- JALANKAN PROGRAM ---
color_segmentation_hsv(INPUT_FILENAME, HUE_MIN, HUE_MAX)
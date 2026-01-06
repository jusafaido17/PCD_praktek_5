import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- FUNGSI KIRSCH DAN MANUAL FILTER (TETAP SAMA) ---

def apply_manual_filter(image, kernel_x, kernel_y):
    """Menerapkan filter konvolusi manual untuk Roberts/Prewitt/Sobel."""
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, edged = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edged

def kirsch_compass_operator(image):
    """Menerapkan Operator Kirsch Compass (8 Arah)."""
    kernels = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32), # K0 (E)
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32), # K1 (NE)
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32), # K2 (N)
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32), # K3 (NW)
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32), # K4 (W)
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32), # K5 (SW)
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32), # K6 (S)
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32), # K7 (SE)
    ]
    responses = [np.abs(cv2.filter2D(image, cv2.CV_64F, k)) for k in kernels]
    max_response = np.maximum.reduce(responses)
    max_response = cv2.normalize(max_response, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, edged = cv2.threshold(max_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edged

# --------------------------------------------------------------------------------

def edge_detection_praktek_5(filename):
    
    img_bgr = cv2.imread(filename)
    if img_bgr is None:
        print(f"❌ ERROR: Tidak dapat memuat gambar: {filename}")
        print("Pastikan file gambar ada di folder yang sama.")
        return

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- HITUNG SEMUA OPERATOR (Sama seperti sebelumnya) ---
    kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    E_roberts = apply_manual_filter(img_gray, kernel_roberts_x, kernel_roberts_y)
    
    kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    E_prewitt = apply_manual_filter(img_gray, kernel_prewitt_x, kernel_prewitt_y)

    E_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    E_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    E_sobel_mag = np.sqrt(E_sobel_x**2 + E_sobel_y**2) 
    E_sobel = cv2.normalize(E_sobel_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    E_sobel_edged = cv2.threshold(E_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    E_kirsch = kirsch_compass_operator(img_gray)

    kernel_laplacian = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]], dtype=np.float32)
    I_laplacian = cv2.filter2D(img_gray, cv2.CV_64F, kernel_laplacian)
    I_laplacian = cv2.normalize(I_laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, E_laplacian = cv2.threshold(I_laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    E_canny = cv2.Canny(img_gray, 100, 200) 
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10)) 
    axes_flat = axes.flat
    
    plt.suptitle('Praktek 5: Deteksi Tepi', fontsize=18) 
    
    images = [
        img_gray, E_roberts, E_prewitt, E_sobel_edged, 
        E_kirsch, E_laplacian, E_canny
    ]

    titles = [
        f'1. Citra Asli',
        '2. Roberts (Gradien Dasar)',
        '3. Prewitt (Gradien Dasar)',
        '4. Sobel (Gradien/Isotropic)',
        '5. Kirsch (Gradien/Compass)',
        '6. Laplacian (Orde Kedua)',
        '7. Canny (Multitahap)'
    ]
    
    for i in range(len(images)):
        ax = axes_flat[i]
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i], fontsize=14, fontweight='bold') 
        ax.axis('off')

    axes_flat[7].set_visible(False) 

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.3)
    plt.show()

# --- PANGGIL FUNGSI ---
input_file = 'cars_25.tif' 
edge_detection_praktek_5(input_file)
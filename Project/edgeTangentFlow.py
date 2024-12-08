import cv2
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def compute_gradients(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return gradient_x, gradient_y

def compute_structure_tensor(gradient_x, gradient_y):
    Jxx = gradient_x ** 2
    Jyy = gradient_y ** 2
    Jxy = gradient_x * gradient_y
    return Jxx, Jyy, Jxy

def gaussian_blur(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def compute_etf(Jxx, Jyy, Jxy):
    print("calculating etf")
    print("jxx",Jxx.shape)
    
    etf_map = np.zeros_like(Jxx)
    
    for i in tqdm (range(Jxx.shape[0])):
        for j in range(Jxx.shape[1]):
            J = np.array([[Jxx[i, j], Jxy[i, j]], [Jxy[i, j], Jyy[i, j]]])
            eigenvalues, eigenvectors = np.linalg.eig(J)
            idx_min_ev = np.argmin(eigenvalues)
            etf_map[i, j] = np.arctan2(eigenvectors[1, idx_min_ev], eigenvectors[0, idx_min_ev])
    return etf_map

def anisotropic_gaussian_blur(image, etf_map, kernel_size, sigma):
    blurred_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            angle = etf_map[i, j]
            kernel = cv2.getGaussianKernel(kernel_size, sigma, cv2.CV_32F)
            rotated_kernel = cv2.warpAffine(kernel.reshape(1, kernel_size), cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle * 180 / np.pi, 1.0), (kernel_size, kernel_size))
            blurred_image[i, j] = cv2.filter2D(image, cv2.CV_32F, rotated_kernel)[i, j]
    return blurred_image
# Load the image
image = load_image('blabla.png')

# Compute gradients
gradient_x, gradient_y = compute_gradients(image)

# Compute structure tensor
Jxx, Jyy, Jxy = compute_structure_tensor(gradient_x, gradient_y)

# Apply Gaussian blur to structure tensor components
kernel_size = (0, 0)
sigma = 2
Jxx_blurred = gaussian_blur(Jxx, kernel_size, sigma)
Jyy_blurred = gaussian_blur(Jyy, kernel_size, sigma)
Jxy_blurred = gaussian_blur(Jxy, kernel_size, sigma)

# Compute ETF map
etf_map = compute_etf(Jxx_blurred, Jyy_blurred, Jxy_blurred)

# Apply anisotropic Gaussian blur along the ETF directions
kernel_size = 31  # Adjust this value as needed
sigma = 5.0  # Adjust this value as needed
# blurred_image = anisotropic_gaussian_blur(image, etf_map, kernel_size, sigma)

# Visualize the ETF map
plt.figure(figsize=(10, 8))
plt.imshow(etf_map, cmap='hsv')
plt.colorbar()
plt.title('Edge Tangent Flow (ETF) Map')
plt.show()
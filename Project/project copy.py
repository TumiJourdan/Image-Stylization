import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Load the image
image = cv2.imread('jenna1080.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not load image")
else:
    # Parameters
    std = 2.28
    k = 1.6
    p = 18
    epsilon = 0.6  # Threshold parameter
    phi = 0.60  # Sensitivity parameter for the continuous ramp--

    # Apply Gaussian blur
    filter_size = 0
    
    g = cv2.GaussianBlur(image, (filter_size, filter_size), std)
    gk = cv2.GaussianBlur(image, (filter_size, filter_size), k*std)
    
    # Compute the Difference of Gaussians
    difference_of_gaussians = np.subtract((1+p)*g,p*gk)

    # Apply the continuous ramp thresholding function
    def continuous_ramp(u, epsilon, phi):
        return np.where(u >= epsilon, 1, 1 + np.tanh(phi * (u - epsilon)))
    # 1 + np.tanh(phi * (u - epsilon))
    # Normalize the DoG result to a range [0, 1] before applying threshold
    
    norm_dog = cv2.normalize(difference_of_gaussians, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Apply the continuous ramp function
    xdog_result = continuous_ramp(norm_dog, epsilon, phi)

    # Display the result
    plt.imshow(xdog_result, cmap='gray')
    plt.axis('off')  # Remove the axes
    plt.show()
    
# STRUCTURE TENSOR ---------------

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

def smooth_tensor(Jxx, Jyy, Jxy, sigma):
    Sxx = cv2.GaussianBlur(Jxx, (0, 0), sigma)
    Syy = cv2.GaussianBlur(Jyy, (0, 0), sigma)
    Sxy = cv2.GaussianBlur(Jxy, (0, 0), sigma)
    return Sxx, Syy, Sxy

def compute_eigenvectors(Bxx, Byy, Bxy):
    height, width = Bxx.shape
    eigenvectors = np.zeros((height, width, 2))

    for y in range(height):
        for x in range(width):
            J = np.array([[Bxx[y, x], Bxy[y, x]],
                          [Bxy[y, x], Byy[y, x]]])
            eigvals, eigvecs = np.linalg.eigh(J)
            eigenvectors[y, x] = eigvecs[:, np.argmin(eigvals)]  # Eigenvector corresponding to the smallest eigenvalue

    return eigenvectors

def blur_across_edges(image, eigenvectors, sigma_blur):
    height, width = image.shape
    rotated_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            vx, vy = eigenvectors[y, x]
            angle = np.arctan2(vy, vx)
            rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), np.degrees(angle), 1)
            rotated_image[y, x] = image[y, x]

    rotated_blurred_image = cv2.GaussianBlur(rotated_image, (0, 0), sigma_blur)

    blurred_image = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            print(x,y)
            vx, vy = eigenvectors[y, x]
            angle = np.arctan2(vy, vx)
            rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), -np.degrees(angle), 1)
            blurred_image[y, x] = cv2.warpAffine(rotated_blurred_image, rotation_matrix, (width, height))[y, x]
    return blurred_image

# Example usage:
image_path = 'jenna1080.jpg'  # Replace with your image path
image = load_image(image_path)
gradient_x, gradient_y = compute_gradients(image)
Jxx, Jyy, Jxy = compute_structure_tensor(gradient_x, gradient_y)
sigma_structure = 2  # Standard deviation for Gaussian smoothing of structure tensor
Sxx, Syy, Sxy = smooth_tensor(Jxx, Jyy, Jxy, sigma_structure)

# Compute eigenvectors
eigenvectors = compute_eigenvectors(Sxx, Syy, Sxy)

# Parameters for blurring across edges
sigma_blur = 2  # Standard deviation for Gaussian blur across edges

# Blur the image across the edges
sigma_e = 2.28
k = 1.6
p = 18
epsilon = 0.6  # Threshold parameter
phi = 0.60  # Sensitivity parameter for the continuous ramp--

# Apply Gaussian blur
filter_size = 0

g = blur_across_edges(image, eigenvectors, sigma_e)
gk = blur_across_edges(image, eigenvectors, k*sigma_e)

# Compute the Difference of Gaussians
difference_of_gaussians = np.subtract((1+p)*g,p*gk)

# Apply the continuous ramp thresholding function
def continuous_ramp(u, epsilon, phi):
    return np.where(u >= epsilon, 1, 1 + np.tanh(phi * (u - epsilon)))
# 1 + np.tanh(phi * (u - epsilon))
# Normalize the DoG result to a range [0, 1] before applying threshold

norm_dog = cv2.normalize(difference_of_gaussians, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Apply the continuous ramp function
xdog_result = continuous_ramp(norm_dog, epsilon, phi)

# Display the result
plt.imshow(xdog_result, cmap='gray')
plt.axis('off')  # Remove the axes
plt.show()
    
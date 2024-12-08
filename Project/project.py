import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('jenna1080.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not load image")
else:
    # Parameters
    std = 2.28
    k = 1.6
    p = 13
    epsilon = 0.6  # Threshold parameter
    phi = 0.60  # Sensitivity parameter for the continuous ramp--

    # Apply Gaussian blur
    filter_size = 15
    
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
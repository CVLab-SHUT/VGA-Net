import numpy as np
from scipy.signal import convolve2d
import cv2

# Global variables for convolution kernels
kernel_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
kernel_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

def l1_l0_minimization(I, mu1, mu2, rho0, max_iterations=100, tolerance=1e-5):
    # Initialize variables
    I_s = np.zeros_like(I)  # Smooth layer
    G1 = np.zeros_like(I)  # Auxiliary variable for smooth layer
    G2 = np.zeros_like(I)  # Auxiliary variable for detail layer
    lamb1 = np.zeros_like(I)  # Lagrange multiplier for smooth layer
    lamb2 = np.zeros_like(I)  # Lagrange multiplier for detail layer
    I_s_old = np.zeros_like(I)  # Previous smooth layer
    
    # ADMM iterations
    for iter in range(max_iterations):
        # Update smooth layer
        I_s = update_smooth_layer(I, I_s, G1, G2, lamb1, rho0, mu1, mu2)
        
        # Update auxiliary variables and Lagrange multipliers
        G1, G2, lamb1, lamb2 = update_auxiliary_variables(I, I_s, G1, G2, lamb1, lamb2, rho0)
        
        # Check convergence
        diff = np.linalg.norm(I_s - I_s_old) / np.linalg.norm(I_s)
        if diff < tolerance:
            break
        
        I_s_old = I_s
    
    # Compute detail layer
    I_detail = I - I_s
    
    return I_detail

def update_smooth_layer(I, I_s, G1, G2, lamb1, rho0, mu1, mu2):
    # Update smooth layer using ADMM
    
    # Compute gradient of smooth layer
    grad_I_s_x = convolve2d(I_s, kernel_x, mode='same', boundary='wrap')
    grad_I_s_y = convolve2d(I_s, kernel_y, mode='same', boundary='wrap')
    
    # Update smooth layer using shrinkage operator
    term1 = I + grad_I_s_x * rho0
    term2 = I + grad_I_s_y * rho0
    I_s = np.maximum(np.sqrt(term1**2 + term2**2) - mu1 / rho0, 0) * np.sign(I)
    
    return I_s

def update_auxiliary_variables(I, I_s, G1, G2, lamb1, lamb2, rho0):
    # Update auxiliary variables and Lagrange multipliers
    
    # Compute gradients
    grad_I_s_x = convolve2d(I_s, kernel_x, mode='same', boundary='wrap')
    grad_I_s_y = convolve2d(I_s, kernel_y, mode='same', boundary='wrap')
    grad_I_x = convolve2d(I, kernel_x, mode='same', boundary='wrap')
    grad_I_y = convolve2d(I, kernel_y, mode='same', boundary='wrap')
    
    # Update G1 and G2
    G1 = grad_I_s_x + lamb1 / rho0
    G2 = grad_I_x - grad_I_s_x + lamb2 / rho0
    
    # Update Lagrange multipliers
    lamb1 = lamb1 + rho0 * (grad_I_s_x - G1)
    lamb2 = lamb2 + rho0 * ((grad_I_x - grad_I_s_x) - G2)
    
    return G1, G2, lamb1, lamb2



# Load retina image
retina_image = cv2.imread('Honeyview_im0139.jpg', cv2.IMREAD_GRAYSCALE)

# Set parameters
mu1 = 0.5
mu2 = 0.005
rho0 = 2

# Apply l1_l0_minimization algorithm
enhanced_retina_image = l1_l0_minimization(retina_image, mu1, mu2, rho0)

# Display original and enhanced images
cv2.imshow('Original Retina Image', retina_image)
cv2.imshow('Honeyview_im0139_enhanced', enhanced_retina_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

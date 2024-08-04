import numpy as np
from PIL import Image

def compute_linf_score(original_image_path, perturbed_image_path):
    # Load images
    original_image = Image.open(original_image_path)
    perturbed_image = Image.open(perturbed_image_path)

    # Resize images to 36x36
    original_image = original_image.resize((36, 36), Image.LANCZOS)
    perturbed_image = perturbed_image.resize((36, 36), Image.LANCZOS)

    # Convert images to numpy arrays
    original_image = np.array(original_image).astype(np.float32)
    perturbed_image = np.array(perturbed_image).astype(np.float32)

    # Normalize images to the range [0, 1]
    original_image /= 255.0
    perturbed_image /= 255.0
    
    # Ensure the images have the same shape
    if original_image.shape != perturbed_image.shape:
        raise ValueError("Images must have the same dimensions and number of channels")
    
    # Compute the absolute difference between the images
    difference = np.abs(original_image - perturbed_image)
    
    # Compute the maximum absolute difference (L-infinity norm)
    linf_score = np.max(np.abs(difference))
    
    return linf_score

print("L_inf Norm ---------------------------------------------------------------------------")
print(f"Image shape : {np.array(Image.open('aeroplane0.jpeg')).astype(np.float32).shape}")
print(f'Estimated epsilon for high noise (L-infinity norm): {compute_linf_score("aeroplane0.jpeg", "aeroplane0-noise-high.jpeg")}')
print(f'Estimated epsilon for low noise (L-infinity norm): {compute_linf_score("aeroplane0.jpeg", "aeroplane0-noise-low.jpeg")}')
print(f'Estimated epsilon for high blur (L-infinity norm): {compute_linf_score("aeroplane0.jpeg", "aeroplane0-zoomblur-high.jpeg")}')
print(f'Estimated epsilon for low blur (L-infinity norm): {compute_linf_score("aeroplane0.jpeg", "aeroplane0-zoomblur-low.jpeg")}')
print(f'Estimated epsilon for darker image (L-infinity norm): {compute_linf_score("aeroplane0.jpeg", "aeroplane0-darker.jpeg")}')




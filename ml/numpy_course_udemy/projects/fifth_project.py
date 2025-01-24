import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
 
def load_image(image_path):
    """Load an image and convert it to a NumPy array."""
    image = Image.open(image_path)
    return np.array(image)
 
def display_image(image_array, title):
    """Display a NumPy array as an image."""
    plt.imshow(image_array)
    plt.title(title)
    plt.axis('off')
 
def flip_image(image_array, mode='horizontal'):
    """Flip the image either horizontally or vertically."""
    if mode == 'horizontal':
        return np.fliplr(image_array)
    elif mode == 'vertical':
        return np.flipud(image_array)
    else:
        raise ValueError("Mode must be 'horizontal' or 'vertical'.")
 
def rotate_image(image_array, degrees):
    """Rotate the image by 90, 180, or 270 degrees."""
    if degrees == 90:
        return np.rot90(image_array, k=1)
    elif degrees == 180:
        return np.rot90(image_array, k=2)
    elif degrees == 270:
        return np.rot90(image_array, k=3)
    else:
        raise ValueError("Degrees must be 90, 180, or 270.")
 
def convert_to_grayscale(image_array):
    """Convert an image to grayscale."""
    return np.mean(image_array, axis=2, dtype=int)
 
def adjust_brightness(image_array, factor):
    """Adjust the brightness of an image."""
    brightened_image = np.clip(image_array * factor, 0, 255)
    return brightened_image.astype(np.uint8)
 
def main():
    print("Image Manipulation with NumPy")
    image_path = input("Enter the path to the image file: ")
    
    # Load the image
    image_array = load_image(image_path)
    
    # Display the original image
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    display_image(image_array, "Original Image")
    
    # Flip the image horizontally
    flipped_horiz = flip_image(image_array, mode='horizontal')
    plt.subplot(2, 3, 2)
    display_image(flipped_horiz, "Flipped Horizontally")
    
    # Flip the image vertically
    flipped_vert = flip_image(image_array, mode='vertical')
    plt.subplot(2, 3, 3)
    display_image(flipped_vert, "Flipped Vertically")
    
    # Rotate the image 90 degrees
    rotated_90 = rotate_image(image_array, 90)
    plt.subplot(2, 3, 4)
    display_image(rotated_90, "Rotated 90 Degrees")
    
    # Convert the image to grayscale
    grayscale_image = convert_to_grayscale(image_array)
    plt.subplot(2, 3, 5)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    
    # Adjust the brightness
    brightened_image = adjust_brightness(image_array, factor=1.5)
    plt.subplot(2, 3, 6)
    display_image(brightened_image, "Brightened Image")
    
    plt.tight_layout()
    plt.show()
 
if __name__ == "__main__":
    main()
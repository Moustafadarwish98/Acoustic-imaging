import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, minimum_filter, maximum_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import io
from scipy.signal import convolve2d

def plot_images(original, reconstructed, title):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image with applied noise')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.title(f'Reconstructed Image - {title}')

    plt.show()

def apply_matched_filter(image):
    # Define a simple template (you may need a more meaningful template)
    template = np.array([[1, 1, 1],
                         [1, 2, 1],
                         [1, 1, 1]])

    # Normalize the template
    template = template / np.sum(template)

    # Apply convolution with the template to each channel
    matched_filtered_image = np.zeros_like(image, dtype=float)
    for channel in range(image.shape[-1]):
        matched_filtered_image[..., channel] = convolve2d(image[..., channel], template, mode='same', boundary='symm', fillvalue=0)

    return matched_filtered_image.astype(np.uint8)

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Read an input image
    input_image_path = 'image4.jpeg'  # Change this to the path of your image
    original_image = io.imread(input_image_path).astype(np.uint8)

    # Introduce random defects at random positions
    num_defects = 100
    defect_positions = np.random.choice(min(original_image.shape[:-1]), (num_defects, 2), replace=True)
    defect_size = 5

    for defect_pos in defect_positions:
        original_image[defect_pos[0]:defect_pos[0] + defect_size, defect_pos[1]:defect_pos[1] + defect_size, :] = 0

    # Add some random noise to the image
    original_image = (original_image + np.random.normal(scale=10, size=original_image.shape)).astype(np.uint8)

    # Apply matched filtering
    matched_filtered_image = apply_matched_filter(original_image)

    # Apply median filtering
    median_filtered_image = median_filter(original_image, size=3)

    # Apply Max-Min filtering
    min_filtered_image = minimum_filter(original_image, size=3)
    max_filtered_image = maximum_filter(original_image, size=3)
    max_min_filtered_image = (min_filtered_image + max_filtered_image) / 2

    # Calculate PSNR and SSIM for each method
    psnr_matched = peak_signal_noise_ratio(original_image, matched_filtered_image, data_range=original_image.max() - original_image.min())
    ssim_matched = structural_similarity(original_image, matched_filtered_image, win_size=5,channel_axis=-1, multichannel=True, data_range=original_image.max() - original_image.min())

    psnr_median = peak_signal_noise_ratio(original_image, median_filtered_image, data_range=original_image.max() - original_image.min())
    ssim_median = structural_similarity(original_image, median_filtered_image, win_size=5,channel_axis=-1, multichannel=True, data_range=original_image.max() - original_image.min())

    psnr_max_min = peak_signal_noise_ratio(original_image, max_min_filtered_image, data_range=original_image.max() - original_image.min())
    ssim_max_min = structural_similarity(original_image, max_min_filtered_image, win_size=5,channel_axis=-1, multichannel=True, data_range=original_image.max() - original_image.min())

    # Print quantitative results
    print(f'PSNR - Matched Filter: {psnr_matched:.2f}, SSIM: {ssim_matched:.2f}')
    print(f'PSNR - Median Filter: {psnr_median:.2f}, SSIM: {ssim_median:.2f}')
    print(f'PSNR - Max-Min Filter: {psnr_max_min:.2f}, SSIM: {ssim_max_min:.2f}')

    # Plot original and reconstructed images
    plot_images(original_image, matched_filtered_image, 'Matched Filter')
    plot_images(original_image, median_filtered_image, 'Median Filter')
    plot_images(original_image, max_min_filtered_image, 'Max-Min Filter')

if __name__ == "__main__":
    main()

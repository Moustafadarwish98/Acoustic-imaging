import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, convolve
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import time
k = 2
"""Used"""

def generate_acoustic_signal(length, num_defects, defect_size, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)

    # Generate a random acoustic signal
    acoustic_signal = np.random.randn(length)

    # Introduce random defects at fixed positions with random amplitudes
    defect_positions = np.random.choice(length, num_defects, replace=False)
    np.random.seed(seed)  # Reset seed for the same defect positions
    for defect_pos in defect_positions:
        defect_end = min(defect_pos + defect_size, length)
        defect_amplitude = np.random.uniform(0.5, 2.0)  # Random defect amplitude
        acoustic_signal[defect_pos:defect_end] *= defect_amplitude

    return acoustic_signal, defect_positions

def apply_acoustic_filter(signal, filter_type='median', kernel_size=k):
    # Apply acoustic-specific filters
    start_time = time.time()

    if filter_type == 'median':
        filtered_signal = medfilt(signal, kernel_size=kernel_size)
    elif filter_type == 'max_min':
        start_time = time.time()
        min_filtered_signal = minimum_filter1d(signal, size=kernel_size)
        max_filtered_signal = maximum_filter1d(signal, size=kernel_size)
        filtered_signal = (min_filtered_signal + max_filtered_signal) / 2
    elif filter_type == 'matched':
        # Design a matched filter based on the expected characteristics (replace with the actual expected signal)
        expected_signal = np.array([1, 1, 1])
        matched_filter = np.flip(expected_signal)
        filtered_signal = convolve(signal, matched_filter, mode='same')
    else:
        # Add more acoustic filters as needed
        raise ValueError(f"Unsupported filter type: {filter_type}")

    elapsed_time = time.time() - start_time
    return filtered_signal, elapsed_time

def calculate_acoustic_metrics(original, filtered):
    # Calculate acoustic-specific metrics
    data_range = np.max(original) - np.min(original)
    psnr_value = peak_signal_noise_ratio(original, filtered, data_range=data_range)
    ssim_value, _ = structural_similarity(original, filtered, full=True, data_range=np.max(original))
    mse_value = mean_squared_error(original, filtered)
    return psnr_value, ssim_value, mse_value

def plot_acoustic_signals(time, signal, filtered_signal, defect_pos, title, filter_name="", kernel_size=k):
    # Plot acoustic signals with defects
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, signal, label='Original Acoustic Signal')
    plt.plot(defect_pos, signal[defect_pos], 'ro', label='Defects')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Original Acoustic Signal with Defects - {title}')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_signal, label=f'Filtered Acoustic Signal ({filter_name}, Kernel Size: {kernel_size})',
             color='green')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Filtered Acoustic Signal - {title}')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# Parameters for the acoustic signal
signal_length = 1000
num_defects = 10
defect_size = 10

# Generate fixed random acoustic signal with defects and random amplitudes
acoustic_signal, defect_positions = generate_acoustic_signal(signal_length, num_defects, defect_size)

# Apply acoustic filtering - Median Filter
acoustic_filtered_median, elapsed_time_median = apply_acoustic_filter(acoustic_signal, filter_type='median', kernel_size=7)

# Calculate metrics and computational time for acoustic signals - Median Filter
psnr_median, ssim_median, mse_median = calculate_acoustic_metrics(acoustic_signal, acoustic_filtered_median)
print("Median Filter Metrics:")
print(f"PSNR: {psnr_median}, SSIM: {ssim_median}, MSE: {mse_median}")
print(f"Computational Time: {elapsed_time_median} seconds")

# Generate time values for plotting acoustic signals - Median Filter
time_acoustic = np.arange(len(acoustic_signal))

# Plot the acoustic signals and defects for the median filter
plot_acoustic_signals(time_acoustic, acoustic_signal, acoustic_filtered_median, defect_positions, 'Median Filter',
                      'Median Filter', kernel_size=k)

# Apply acoustic filtering - Max-Min Filter
acoustic_filtered_max_min, elapsed_time_max_min = apply_acoustic_filter(acoustic_signal, filter_type='max_min', kernel_size=k)

# Calculate metrics and computational time for acoustic signals - Max-Min Filter
psnr_max_min, ssim_max_min, mse_max_min = calculate_acoustic_metrics(acoustic_signal, acoustic_filtered_max_min)
print("\nMax-Min Filter Metrics:")
print(f"PSNR: {psnr_max_min}, SSIM: {ssim_max_min}, MSE: {mse_max_min}")
print(f"Computational Time: {elapsed_time_max_min} seconds")

# Generate time values for plotting acoustic signals - Max-Min Filter
time_acoustic = np.arange(len(acoustic_signal))

# Plot the acoustic signals and defects for the max-min filter
plot_acoustic_signals(time_acoustic, acoustic_signal, acoustic_filtered_max_min, defect_positions, 'Max-Min Filter',
                      'Max-Min Filter', kernel_size=k)

# Apply acoustic filtering - Matched Filter
acoustic_filtered_matched, elapsed_time_matched = apply_acoustic_filter(acoustic_signal, filter_type='matched', kernel_size=7)

# Calculate metrics and computational time for acoustic signals - Matched Filter
psnr_matched, ssim_matched, mse_matched = calculate_acoustic_metrics(acoustic_signal, acoustic_filtered_matched)
print("\nMatched Filter Metrics:")
print(f"PSNR: {psnr_matched}, SSIM: {ssim_matched}, MSE: {mse_matched}")
print(f"Computational Time: {elapsed_time_matched} seconds")

# Generate time values for plotting acoustic signals - Matched Filter
time_acoustic = np.arange(len(acoustic_signal))

# Plot the acoustic signals and defects for the matched filter
plot_acoustic_signals(time_acoustic, acoustic_signal, acoustic_filtered_matched, defect_positions, 'Matched Filter',
                      'Matched Filter', kernel_size=k)

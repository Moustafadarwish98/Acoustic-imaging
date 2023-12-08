import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import time
"""used threshold """
def generate_acoustic_signal(length, amplitude_factor=1.0, noise_amplitude=0.2):
    original_signal = amplitude_factor * np.random.randn(length)
    noise_signal = noise_amplitude * np.random.randn(length)
    combined_signal = original_signal + noise_signal
    return combined_signal, original_signal, noise_signal

def apply_acoustic_filter(signal, filter_type='median', kernel_size=3, threshold=None):
    if threshold is not None:
        signal[np.abs(signal) < np.abs(threshold)] = 0

    if filter_type == 'median':
        filtered_signal = medfilt(signal, kernel_size=kernel_size)
    elif filter_type == 'max_min':
        min_filtered_signal = minimum_filter1d(signal, size=kernel_size)
        max_filtered_signal = maximum_filter1d(signal, size=kernel_size)
        filtered_signal = (min_filtered_signal + max_filtered_signal) / 2
    elif filter_type == 'matched':
        filtered_signal = signal.copy()
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    return filtered_signal

def calculate_acoustic_metrics(original, filtered):
    data_range = np.max(original) - np.min(original)
    psnr_value = peak_signal_noise_ratio(original, filtered, data_range=data_range)
    ssim_value, _ = structural_similarity(original, filtered, full=True, data_range=np.max(original))
    mse_value = mean_squared_error(original, filtered)
    return psnr_value, ssim_value, mse_value

def plot_acoustic_signals(time, original_signal, noise_signal, filtered_signal, filter_name="", kernel_size=None,
                          threshold=None):
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, original_signal, label='Original Acoustic Signal', color='b')
    plt.plot(time, noise_signal, label='Noise Signal', color='r', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Original Acoustic Signal with Noise - {filter_name}')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_signal, label=f'Filtered Acoustic Signal ({filter_name}, Kernel Size: {kernel_size})',
             color='green')
    if threshold is not None:
        plt.axhline(threshold, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Filtered Acoustic Signal - {filter_name}')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Parameters
signal_length = 2000
amplitude_factor = 3.0
noise_amplitude = 0.4
threshold_value =0.8

# Generate signals
combined_signal, original_signal, noise_signal = generate_acoustic_signal(
    signal_length, amplitude_factor, noise_amplitude
)

# Plot original signal with noise
plt.figure(figsize=(15, 3))
plt.plot(original_signal, label='Original Acoustic Signal', color='b')
plt.plot(noise_signal, label='Noise Signal', color='r', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Acoustic Signal with Noise')
plt.legend()
plt.show()

# Filter with Median Filter and plot
start_time = time.time()
acoustic_filtered_median = apply_acoustic_filter(combined_signal, filter_type='median', kernel_size=3,
                                                 threshold=threshold_value)
end_time = time.time()
plot_acoustic_signals(np.arange(len(acoustic_filtered_median)), original_signal, noise_signal,
                      acoustic_filtered_median, 'Median Filter', kernel_size=3, threshold=threshold_value)
psnr_median, ssim_median, mse_median = calculate_acoustic_metrics(original_signal, acoustic_filtered_median)
print("\nMedian Filter Metrics:")
print(f"PSNR: {psnr_median}, SSIM: {ssim_median}, MSE: {mse_median}")
print(f"Computational Time: {end_time - start_time} seconds")

# Filter with Max-Min Filter and plot
start_time = time.time()
acoustic_filtered_max_min = apply_acoustic_filter(combined_signal, filter_type='max_min', kernel_size=5,
                                                  threshold=threshold_value)
end_time = time.time()
plot_acoustic_signals(np.arange(len(acoustic_filtered_max_min)), original_signal, noise_signal,
                      acoustic_filtered_max_min, 'Max-Min Filter', kernel_size=5, threshold=threshold_value)
psnr_max_min, ssim_max_min, mse_max_min = calculate_acoustic_metrics(original_signal, acoustic_filtered_max_min)
print("\nMax-Min Filter Metrics:")
print(f"PSNR: {psnr_max_min}, SSIM: {ssim_max_min}, MSE: {mse_max_min}")
print(f"Computational Time: {end_time - start_time} seconds")

# Filter with Matched Filter and plot
start_time = time.time()
acoustic_filtered_matched = apply_acoustic_filter(combined_signal, filter_type='matched', threshold=threshold_value)
end_time = time.time()
plot_acoustic_signals(np.arange(len(acoustic_filtered_matched)), original_signal, noise_signal,
                      acoustic_filtered_matched, 'Matched Filter', threshold=threshold_value)
psnr_matched, ssim_matched, mse_matched = calculate_acoustic_metrics(original_signal, acoustic_filtered_matched)
print("\nMatched Filter Metrics:")
print(f"PSNR: {psnr_matched}, SSIM: {ssim_matched}, MSE: {mse_matched}")
print(f"Computational Time: {end_time - start_time} seconds")

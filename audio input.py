import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import librosa
import librosa.display
"""audio file"""

def read_audio_file(file_path):
    # Read audio file using librosa
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr

def apply_acoustic_filter(signal, filter_type='median', kernel_size=3):
    # Apply acoustic-specific filters
    if filter_type == 'median':
        filtered_signal = medfilt(signal, kernel_size=kernel_size)
    elif filter_type == 'max_min':
        min_filtered_signal = minimum_filter1d(signal, size=kernel_size)
        max_filtered_signal = maximum_filter1d(signal, size=kernel_size)
        filtered_signal = (min_filtered_signal + max_filtered_signal) / 2
    elif filter_type == 'matched':
        # Matched filter (no actual filtering, just a copy of the original signal)
        filtered_signal = signal.copy()
    else:
        # Add more acoustic filters as needed
        raise ValueError(f"Unsupported filter type: {filter_type}")

    return filtered_signal

def calculate_acoustic_metrics(original, filtered):
    # Calculate acoustic-specific metrics
    data_range = np.max(original) - np.min(original)
    psnr_value = peak_signal_noise_ratio(original, filtered, data_range=data_range)
    ssim_value, _ = structural_similarity(original, filtered, full=True, data_range=np.max(original))
    mse_value = mean_squared_error(original, filtered)
    return psnr_value, ssim_value, mse_value

def plot_acoustic_signals(time, signal, filtered_signal, defect_pos, title, filter_name="", kernel_size=None):
    # Plot acoustic signals with defects
    plt.figure(figsize=(15, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label='Original Acoustic Signal')
    plt.plot(defect_pos, signal[defect_pos], 'ro', label='Defects')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Original Acoustic Signal with Defects - {title}')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, filtered_signal, label=f'Filtered Acoustic Signal ({filter_name}, Kernel Size: {kernel_size})',
             color='green')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Filtered Acoustic Signal - {title}')
    plt.legend(loc='upper left')

    # Plot spectrogram
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(signal), ref=np.max), x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram with Defects')

    # Highlight defect positions
    for pos in defect_pos:
        plt.vlines(librosa.frames_to_time(pos), 0, sr // 2, colors='r', linestyles='dashed')

    plt.tight_layout()
    plt.show()

# Specify the path to your audio file
audio_file_path = 'path/to/your/audio/file.wav'

# Read audio file
acoustic_signal, sr = read_audio_file(audio_file_path)

# Apply acoustic filtering - Median Filter
acoustic_filtered_median = apply_acoustic_filter(acoustic_signal, filter_type='median', kernel_size=3)

# Calculate metrics for acoustic signals - Median Filter
psnr_median, ssim_median, mse_median = calculate_acoustic_metrics(acoustic_signal, acoustic_filtered_median)

# Display metrics for acoustic signals - Median Filter
print("Median Filter Metrics:")
print(f"PSNR: {psnr_median}, SSIM: {ssim_median}, MSE: {mse_median}")

# Generate time values for plotting acoustic signals - Median Filter
time_acoustic = np.arange(len(acoustic_signal))

# Plot the acoustic signals and defects for the median filter
plot_acoustic_signals(time_acoustic, acoustic_signal, acoustic_filtered_median, defect_positions, 'Median Filter',
                      'Median Filter', kernel_size=4)

# Apply acoustic filtering - Max-Min Filter
acoustic_filtered_max_min = apply_acoustic_filter(acoustic_signal, filter_type='max_min', kernel_size=5)

# Calculate metrics for acoustic signals - Max-Min Filter
psnr_max_min, ssim_max_min, mse_max_min = calculate_acoustic_metrics(acoustic_signal, acoustic_filtered_max_min)

# Display metrics for acoustic signals - Max-Min Filter
print("\nMax-Min Filter Metrics:")
print(f"PSNR: {psnr_max_min}, SSIM: {ssim_max_min}, MSE: {mse_max_min}")

# Generate time values for plotting acoustic signals - Max-Min Filter
time_acoustic = np.arange(len(acoustic_signal))

# Plot the acoustic signals and defects for the max-min filter
plot_acoustic_signals(time_acoustic, acoustic_signal, acoustic_filtered_max_min, defect_positions, 'Max-Min Filter',
                      'Max-Min Filter', kernel_size=5)

# Apply acoustic filtering - Matched Filter
acoustic_filtered_matched = apply_acoustic_filter(acoustic_signal, filter_type='matched')

# Calculate metrics for acoustic signals - Matched Filter
psnr_matched, ssim_matched, mse_matched = calculate_acoustic_metrics(acoustic_signal, acoustic_filtered_matched)

# Display metrics for acoustic signals - Matched Filter
print("\nMatched Filter Metrics:")
print(f"PSNR: {psnr_matched}, SSIM: {ssim_matched}, MSE: {mse_matched}")

# Generate time values for plotting acoustic signals - Matched Filter
time_acoustic = np.arange(len(acoustic_signal))

# Plot the acoustic signals and defects for the matched filter
plot_acoustic_signals(time_acoustic, acoustic_signal, acoustic_filtered_matched, defect_positions, 'Matched Filter',
                      'Matched Filter')

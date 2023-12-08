import numpy as np
import matplotlib.pyplot as plt

def generate_acoustic_wave(positions, max_amplitude=1.0, num_anomalies=5, anomaly_intensity=2.0):
    # Generate a simple acoustic wave
    wave = max_amplitude * np.sin(positions)

    # Introduce anomalies at random positions with random intensities
    anomaly_positions = np.random.choice(len(positions), num_anomalies, replace=False)
    wave[anomaly_positions] += anomaly_intensity * np.random.randn(num_anomalies)

    return wave, anomaly_positions

def plot_acoustic_wave(positions, wave, anomalies, cmap='viridis'):
    plt.figure(figsize=(10, 6))

    # Plot the original acoustic wave
    plt.plot(positions, wave, label='Wave', linewidth=2)

    # Highlight anomalies with a red cloud
    plt.scatter(positions[anomalies], wave[anomalies], c='red', marker='o', s=150, label='Anomalies')

    plt.xlabel('Position')
    plt.ylabel('Amplitude')
    plt.title('Acoustic Wave with Anomalies')
    plt.legend()

    plt.show()

def main():
    np.random.seed(42)  # For reproducibility
    positions = np.linspace(0, 10, 100)

    # Generate acoustic wave with anomalies
    original_wave, anomaly_positions = generate_acoustic_wave(positions)

    # Plot the acoustic wave with anomalies
    plot_acoustic_wave(positions, original_wave, anomaly_positions)

if __name__ == "__main__":
    main()

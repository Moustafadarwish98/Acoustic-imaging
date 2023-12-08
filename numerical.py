import numpy as np
import matplotlib.pyplot as plt

def generate_acoustic_wave(positions, max_amplitude=1.0, num_anomalies=5, anomaly_intensity=2.0):
    # Generate a simple acoustic wave
    wave = max_amplitude * np.sin(positions)

    # Introduce anomalies at random positions with random intensities
    anomaly_positions = np.random.choice(len(positions), num_anomalies, replace=False)
    wave[anomaly_positions] += anomaly_intensity * np.random.randn(num_anomalies)

    return wave, anomaly_positions

def solve_wave_equation(positions, time_steps, c=1.0):
    # Set up the initial conditions
    dt = 0.01
    dx = positions[1] - positions[0]
    wave = np.zeros((len(positions), time_steps))
    wave[:, 0] = generate_acoustic_wave(positions)[0]

    # Numerical solution using finite difference method
    for t in range(1, time_steps):
        wave[1:-1, t] = 2 * (1 - c**2 * (dt / dx)**2) * wave[1:-1, t-1] - wave[1:-1, t-2] + c**2 * (dt / dx)**2 * (wave[2:, t-1] + wave[:-2, t-1])

    return wave

def plot_acoustic_wave(positions, wave, anomalies, cmap='viridis'):
    plt.figure(figsize=(12, 8))

    # Plot the original acoustic wave
    plt.plot(positions, wave[:, 0], label='Initial Wave', linewidth=2)

    # Solve the wave equation and plot subsequent time steps
    for t in range(1, wave.shape[1]):
        plt.plot(positions, wave[:, t], alpha=0.2, label=f'Time Step {t}')

    # Highlight anomalies with a red cloud
    plt.scatter(positions[anomalies], wave[anomalies, 0], c='red', marker='o', s=150, label='Anomalies')

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

    # Solve the wave equation
    time_steps = 100
    wave_solution = solve_wave_equation(positions, time_steps)

    # Plot the acoustic wave with anomalies and solution
    plot_acoustic_wave(positions, wave_solution, anomaly_positions)

    # Print positions of anomalies
    print("Positions of Anomalies:", anomaly_positions)

if __name__ == "__main__":
    main()

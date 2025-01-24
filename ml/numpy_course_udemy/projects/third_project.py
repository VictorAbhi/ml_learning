import numpy as np
import matplotlib.pyplot as plt

def calculate_statistics(temperatures):
    min_temp = np.min(temperatures)
    max_temp = np.max(temperatures)
    mean_temp = np.mean(temperatures)
    median_temp = np.median(temperatures)
    return min_temp, max_temp, mean_temp, median_temp

def identify_extreme_days(temperatures, threshold):
    extreme_days = np.where(temperatures > threshold)[0] + 1 # make array of (bool values) for temperatures greater than threshold
    return extreme_days

def plot_temperature_trends(days, temperatures):
    plt.figure(figsize=(10, 6))
    plt.plot(days, temperatures, marker='o', color='b')
    plt.axhline(y=np.mean(temperatures), color="r", linestyle="--", label="Mean Temperature")
    
    plt.title("10-Day Temperature Trends")
    plt.xlabel("Day")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.xticks(days)
    plt.show()


def main():
    print("Weather Data Analysis")
    print("Analyzing temperature data for 10 days...")
    
    # Sample temperature data for 10 days
    temperatures = np.array([22.5, 24.0, 19.5, 25.2, 23.8, 26.1, 28.4, 21.7, 24.3, 27.6])
    days = np.arange(1, len(temperatures) + 1)
 
    # Calculate statistics
    mean_temp, median_temp, max_temp, min_temp = calculate_statistics(temperatures)
    print(f"\nTemperature Statistics:")
    print(f"Mean Temperature: {mean_temp:.2f}°C")
    print(f"Median Temperature: {median_temp:.2f}°C")
    print(f"Max Temperature: {max_temp:.2f}°C")
    print(f"Min Temperature: {min_temp:.2f}°C")
 
    # Identify extreme temperature days
    threshold = 26.0  # Example threshold for extreme temperatures
    extreme_days = identify_extreme_days(temperatures, threshold)
    print(f"\nDays with temperatures above {threshold}°C: {extreme_days}")
 
    # Visualize temperature trends
    plot_temperature_trends(days, temperatures)
 
if __name__ == "__main__":
    main()
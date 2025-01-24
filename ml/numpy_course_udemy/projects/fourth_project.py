import numpy as np
import matplotlib.pyplot as plt
 
def random_walk_2d(steps):
    # Generate random steps: -1, 0, or 1 for both x and y directions
    x_steps = np.random.choice([-1, 0, 1], size=steps)
    x_steps = np.random.choice([-1, 0, 1], size=steps)
    y_steps = np.random.choice([-1, 0, 1], size=steps)
    
    # Cumulative sum to calculate position at each step
    x_positions = np.cumsum(x_steps)
    
    y_positions = np.cumsum(y_steps)
    
    return x_positions, y_positions
 
def plot_random_walk(x_positions, y_positions):
    plt.figure(figsize=(8, 8))
    plt.plot(x_positions, y_positions, marker="o", linestyle="-", color="b", label="Random Walk Path")
    plt.scatter(0, 0, color="green", label="Start", zorder=5)  # Start point
    plt.scatter(x_positions[-1], y_positions[-1], color="red", label="End", zorder=5)  # End point
    plt.title("2D Random Walk Simulation")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
    plt.axvline(0, color='black', linewidth=0.5, linestyle="--")
    plt.legend()
    plt.grid()
    plt.show()
 
def main():
    print("2D Random Walk Simulation")
    steps = 100  # Number of steps in the random walk
    print(f"Simulating a random walk with {steps} steps...")
    
    # Generate the random walk
    x_positions, y_positions = random_walk_2d(steps)
    
    # Plot the random walk
    plot_random_walk(x_positions, y_positions)
    
    # Calculate and display final displacement
    final_displacement = np.sqrt(x_positions[-1]**2 + y_positions[-1]**2)
    print(f"\nFinal position: ({x_positions[-1]}, {y_positions[-1]})")
    print(f"Total displacement from origin: {final_displacement:.2f} units")
 
if __name__ == "__main__":
    main()
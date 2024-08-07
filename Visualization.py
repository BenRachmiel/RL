import os

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt

from matplotlib.animation import FuncAnimation, PillowWriter


def visualize(positions, orientations, targets, episode, save_dir="visualizations"):
    """
    Visualizes the agent's path and saves the visualization as a static image with directionality arrows.

    Parameters:
    - positions: List of agent positions at each step.
    - orientations: List of agent orientations at each step.
    - targets: List of target positions.
    - episode: The current episode number for labeling purposes.
    - save_dir: The directory to save the image.
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(save_dir, f"episode_{episode}_{timestamp}.png")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)
    ax.set_title(f'Episode {episode} - Agent Path')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

    # Plot the entire path
    x_data, y_data = zip(*positions)
    ax.plot(x_data, y_data, 'bo-', linewidth=2, label='Agent Path')

    # Plot the targets
    target_x, target_y = zip(*targets)
    ax.plot(target_x, target_y, 'ro', label='Target')

    # Add arrows for directionality every few steps
    for i in range(0, len(positions), 5):  # Adjust step size for arrows as needed
        pos = positions[i]
        orient = orientations[i]
        dx = 0.5 * np.cos(orient)
        dy = 0.5 * np.sin(orient)
        ax.arrow(pos[0], pos[1], dx, dy, head_width=2, head_length=1, fc='blue', ec='blue')

    # Add a legend
    ax.legend()

    # Save the plot as a PNG image
    plt.savefig(image_filename)
    plt.close(fig)

    print(f"Visualization saved as {image_filename}.")

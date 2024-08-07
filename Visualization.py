import os

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt

from matplotlib.animation import FuncAnimation, PillowWriter


def gif_visual(positions, orientations, targets, episode, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    gif_filename = os.path.join(save_dir, f"episode_{episode}_{timestamp}.gif")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(-1, 100)
    ax.set_ylim(-1, 100)
    ax.set_title(f'Episode {episode} - Agent Path')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

    target_x, target_y = zip(*targets)
    target_plot, = ax.plot(target_x[0], target_y[0], 'ro', label='Target')

    agent_path, = ax.plot([], [], 'bo-', label='Agent Path')
    target_plot, = ax.plot([], [], 'ro', label='Target')
    direction_arrow = None

    def init():
        agent_path.set_data([], [])
        target_plot.set_data([], [])
        return agent_path, target_plot

    def update(frame):
        x_data, y_data = zip(*positions[:frame + 1])
        agent_path.set_data(x_data, y_data)

        # Update target position - repeat the position to mimic sequence format
        target_plot.set_data([targets[frame][0]], [targets[frame][1]])
        nonlocal direction_arrow
        if direction_arrow:
            direction_arrow.remove()

        pos = positions[frame]
        orient = orientations[frame]
        dx = 0.5 * np.cos(orient)
        dy = 0.5 * np.sin(orient)
        direction_arrow = ax.arrow(pos[0], pos[1], dx, dy, head_width=2, head_length=2, fc='blue', ec='blue')

        return agent_path, target_plot, direction_arrow

    ani = FuncAnimation(
        fig, update, frames=len(positions), init_func=init, blit=False, interval=100, repeat=False
    )
    ani.save(gif_filename, writer=PillowWriter(fps=5))
    print(f"Visualization saved as {gif_filename}.")

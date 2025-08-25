import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def animate_images(images, path, scan, interval=100, cmap="gray"):
    """Helper function to animate a list of images."""
    if interval <= 0:
        raise ValueError("interval must be a positive integer (milliseconds).")
    if len(images) == 0:
        raise ValueError("images must be a non-empty sequence.")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=80)
    xlims_mm = [v * 1e3 for v in scan.xlims]
    zlims_mm = [v * 1e3 for v in scan.zlims]
    im = ax.imshow(
        np.array(images[0]),
        animated=True,
        cmap=cmap,
        extent=[xlims_mm[0], xlims_mm[1], zlims_mm[1], zlims_mm[0]],
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")

    def update(frame):
        im.set_array(np.array(images[frame]))
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(images),
        blit=True,
        interval=interval,
    )
    plt.close(fig)
    fps = max(1, 1000 // interval)

    ani.save(path, writer="imagemagick", fps=fps)

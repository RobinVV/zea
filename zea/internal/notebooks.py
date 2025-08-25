import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation


def animate_images(images, scan, interval=100, cmap="gray", default_mode="reflect"):
    """Helper function to animate a list of images."""
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
    return HTML(ani.to_jshtml(fps=1000 // interval, embed_frames=True, default_mode=default_mode))

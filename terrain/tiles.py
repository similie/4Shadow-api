import numpy as np


def render_tile(data_slice: np.ndarray, cmap="viridis") -> bytes:
    import matplotlib.pyplot as plt
    from io import BytesIO

    fig = plt.figure(frameon=False)
    fig.set_size_inches(256 / 100, 256 / 100)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data_slice, cmap=cmap)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    return buf.getvalue()

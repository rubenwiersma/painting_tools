import numpy as np
import matplotlib.pyplot as plt
from ..measurements.pigments import basic
from .color import spectral_to_rgb, resample_wavelengths


def show_colors(colors, labels=None, scale=1, axis=False, points=None, normalize=True, cmap='Greys_r'):
    if type(colors) is not list:
        colors = [colors]

    fig, axs = plt.subplots(1, len(colors))
    if len(colors) == 1:
        axs = [axs]
    fig.set_size_inches(3 * len(colors) * scale, 5 * scale)
    for i, color in enumerate(colors):
        if not axis:
            axs[i].set_axis_off()
        color = color[np.newaxis, np.newaxis, :] if len(color.shape) < 3 else color
        axs[i].imshow(color, vmin=None if normalize else 0, vmax=None if normalize else 1, cmap=cmap)
        if labels is not None:
            axs[i].set_title(labels[i])
        if points is not None:
            axs[i].scatter(*points[i], c='white', marker='o')
    plt.show()


def show_spectra(colors, labels=None, scale=1, axis=False, spectra=basic.spectra):
    colors = [spectral_to_rgb(color, wavelengths=spectra) for color in colors]
    show_colors(colors, labels, scale, axis)
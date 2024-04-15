import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    @staticmethod
    def visualize(**images):
        """
        Usage: visualize(dict[label: str, data: numpy.ndarray])
        """

        def show(ax, image, title):
            ax.imshow(image)
            ax.set_title(title)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        fig = plt.figure(figsize=(9, 9))
        axs = fig.subplots(1, len(images.keys()))
        i = 0
        for label, data in images.items():
            show(axs[i], data, label)
            i += 1

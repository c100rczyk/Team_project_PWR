import matplotlib.pyplot as plt


def visualize(anchor, positive, negative):
    # visualize a triplet

    def show(ax, image, title):
        ax.imshow(image)
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))
    axs = fig.subplots(1, 3)

    show(axs[0], anchor, "Anchor")
    show(axs[1], positive, "Positive")
    show(axs[2], negative, "Negative")

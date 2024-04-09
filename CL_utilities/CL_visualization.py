from matplotlib import pyplot as plt
import numpy as np


def visualize(data_to_visualize):
    how_many_samples = 0
    for img1, label in data_to_visualize:
        if how_many_samples < 2:
            # print(img1)
            plt.figure(figsize=(10, 5))

            # -------------------------------------------------------
            plt.subplot(1, 2, 1)
            plt.imshow(np.asarray(img1[0][0]).astype('uint8'))
            plt.title('same kind of data')
            plt.subplot(1, 2, 2)
            plt.imshow(np.asarray(img1[1][0]).astype('uint8'))
            plt.title('')
            plt.show()
            print("Etykieta:", np.asarray(label[0]).astype('uint8'))

            # -------------------------------------------------------
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(np.asarray(img1[0][1]).astype('uint8'))
            plt.title('different type of data')
            plt.subplot(1, 2, 2)
            plt.imshow(np.asarray(img1[1][1]).astype('uint8'))
            plt.title('')
            plt.show()
            print("Etykieta:", np.asarray(label[1]).astype('uint8'))
            how_many_samples += 1
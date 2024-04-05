import numpy as np
import random
import os


class TripletGenerator:
    def __init__(self, datasetPath, max_triplets):
        # create an empty list which will contain the subdirectory
        # names of the `dataset` directory with more than one image
        # in it
        self.fruitNames = list()
        # iterate over the subdirectories in the dataset directory
        for folderName in os.listdir(datasetPath):
            # build the subdirectory name
            absoluteFolderName = os.path.join(datasetPath, folderName)
            # get the number of images in the subdirectory
            numImages = len(os.listdir(absoluteFolderName))
            # if the number of images in the current subdirectory
            # is more than one, append into the `fruitNames` list
            if numImages > 1:
                self.fruitNames.append(absoluteFolderName)
        # create a dictionary of people name to their image names
        self.allFruits = self.generate_all_fruits_dict()
        self.max_triplets = max_triplets

    def generate_all_fruits_dict(self):
        # create an empty dictionary that will be populated with
        # directory names as keys and image names as values
        allFruits = dict()
        # iterate over all the directory names with more than one
        # image in it
        for fruit in self.fruitNames:
            # get all the image names in the current directory
            imageNames = os.listdir(fruit)
            # build the image paths and populate the dictionary
            fruitPhotos = [os.path.join(fruit, imageName) for imageName in imageNames]
            allFruits[fruit] = fruitPhotos
        # return the dictionary
        return allFruits

    def get_next_element(self):
        # create an infinite generator
        for _ in range(0, self.max_triplets):
            # draw an image at random which will be our anchor and
            anchorName = random.choice(self.fruitNames)
            # copy the list of images names and remove the anchor from the list
            temporaryNames = self.fruitNames.copy()
            temporaryNames.remove(anchorName)
            # draw an image at random from the list of images without
            # the anchor, which will act as our negative sample
            negativeName = random.choice(temporaryNames)
            # draw two images from the anchor folder without replacement
            (anchorPhoto, positivePhoto) = np.random.choice(
                a=self.allFruits[anchorName], size=2, replace=False
            )
            # draw an image from the negative folder
            negativePhoto = random.choice(self.allFruits[negativeName])
            # yield the anchor, positive and negative photos
            yield anchorPhoto, positivePhoto, negativePhoto

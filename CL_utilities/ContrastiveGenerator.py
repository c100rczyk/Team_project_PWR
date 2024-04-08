import os
import numpy as np
import random



class ContrastiveGenerator:
    def __init__(self, datasetPath, number_of_pairs):  # number_of_pairs : number of iterations in loop next_get_item()
        # empty list that will contain the subdirectory names of products
        # of the dataset directory with more than one image in it
        self.types_of_products = self.read_types_of_products(datasetPath)
        self.number_of_pairs = number_of_pairs
        # create a dictionary of people name to their image names
        self.allProducts = self.generate_all_products_dict()

    # The function below works correctly in 3 cases:
    # 1) images are only in subfolders
    # 2) images are only in subsubfolders
    # 3) images are both in subfolders and subsubfolders
    def read_types_of_products(self, datasetPath) -> list[str]:
        # subdirectories in the main directory
        types = []
        subdirs_main = [d for d in os.listdir(datasetPath) if os.path.isdir(os.path.join(datasetPath, d))]
        # iterate over subdirectories in the main directory:
        for folderName in subdirs_main:
            subdirs_main_path = os.path.join(datasetPath, folderName)
            # subsubdirectories
            subdirs = [d for d in os.listdir(subdirs_main_path) if os.path.isdir(os.path.join(subdirs_main_path, d))]
            # if there are no subsubdirectories, add the subdirectory name
            if len(subdirs) == 0:
                types.append(subdirs_main_path)
            else:
                # traverse through all subsubdirectories and add them
                for directory in subdirs:
                    types.append(os.path.join(subdirs_main_path, directory))
        return types

    # Buildings paths to specific images
    def generate_all_products_dict(self):
        # create an empty dictionary that will be populated with
        # directory names as keys and image names as values
        all_products = dict()
        # populate with images
        for product in self.types_of_products:
            image_names = os.listdir(product)
            # build the image paths and populate the dictionary
            productsPhotos = [os.path.join(product, imageName) for imageName in image_names]
            all_products[product] = productsPhotos
        print(len(all_products))
        return all_products

    def get_next_element(self):
        for i in range(self.number_of_pairs):
            anchor = random.choice(self.types_of_products)

            # copy the list of products
            temporaryImages = self.types_of_products.copy()
            temporaryImages.remove(anchor)

            # random product from a list of products without anchor
            negativeProduct = random.choice(temporaryImages)

            (anchorProduct, positiveProduct) = np.random.choice(
                a=self.allProducts[anchor],
                size=2,
                replace=False
            )
            # Image from the negative folder
            negativeProduct = random.choice(self.allProducts[negativeProduct])

            yield (anchorProduct, positiveProduct, 0.0)
            yield (anchorProduct, negativeProduct, 1.0)




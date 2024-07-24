import numpy as np
import random
from utilities.data_reader import DataReader


class ContrastiveGenerator:
    def __init__(self, dataset_path, max_iterations):
        self.max_iterations = max_iterations
        self.types_of_products = DataReader.read_types_of_products(dataset_path)
        self.image_classes = DataReader.generate_images_classes_dict(
            self.types_of_products
        )

    def get_next_element(self):
        for i in range(self.max_iterations):
            anchor_product_name = random.choice(self.types_of_products)

            temporary_images_classes = self.types_of_products.copy()
            temporary_images_classes.remove(anchor_product_name)

            negative_name = random.choice(temporary_images_classes)

            (anchor_product, positive_product) = np.random.choice(
                a=self.image_classes[anchor_product_name], size=2, replace=False
            )
            negative_product = random.choice(self.image_classes[negative_name])

            yield anchor_product, positive_product, 0.0
            yield anchor_product, negative_product, 1.0

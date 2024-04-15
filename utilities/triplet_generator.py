import numpy as np
import random
from utilities.data_reader import DataReader


class TripletGenerator:
    def __init__(self, dataset_path, max_iterations):
        self.max_iterations = max_iterations
        self.types_of_products = DataReader.read_types_of_products(dataset_path)
        self.image_classes = DataReader.generate_images_classes_dict(
            self.types_of_products
        )

    def get_next_element(self) -> tuple[int]:
        for _ in range(0, self.max_iterations):
            anchor_product = random.choice(self.types_of_products)
            temporary_products_names = self.types_of_products.copy()
            temporary_products_names.remove(anchor_product)
            negative_name = random.choice(temporary_products_names)

            (anchor_image, positive_image) = np.random.choice(
                a=self.image_classes[anchor_product], size=2, replace=False
            )
            negative_image = random.choice(self.image_classes[negative_name])

            yield anchor_image, positive_image, negative_image

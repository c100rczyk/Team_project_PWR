import numpy as np
import random
from utilities.data_reader import DataReader


class ContrastiveGenerator:
    def __init__(self, dataset_path, max_iterations):
        self.max_iterations = max_iterations
        self.types_of_products = DataReader.read_types_of_products(dataset_path)
        self.dict_all_images_name_path = DataReader.generate_images_classes_dict(
            self.types_of_products
        )

    def get_next_element(self):
        """
        Create positive and negatives pairs of images with appropriate labels
        Returns:
            [
            ./data/VegFru/fru92_images/Training/raspberry/f_01_18_0192.jpg
            ./data/VegFru/fru92_images/Training/raspberry/f_01_18_0463.jpg
            0.0

            ./data/VegFru/fru92_images/Training/wampee/f_01_21_0568.jpg
            ./data/VegFru/fru92_images/Training/golden_melon/f_04_01_0206.jpg
            1.0
            ]
        """
        for i in range(self.max_iterations):
            anchor_product_name = random.choice(self.types_of_products)

            temporary_images_classes = self.types_of_products.copy()
            temporary_images_classes.remove(anchor_product_name)

            negative_name = random.choice(temporary_images_classes)

            (anchor_product, positive_product) = np.random.choice(
                a=self.dict_all_images_name_path[anchor_product_name], size=2, replace=False
            )
            negative_product = random.choice(self.dict_all_images_name_path[negative_name])

            yield anchor_product, positive_product, 0.0
            yield anchor_product, negative_product, 1.0

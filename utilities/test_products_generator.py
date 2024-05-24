from utilities.data_reader import DataReader


class TestProductsGenerator:
    def __init__(self, dataset_path):
        self.types_of_products = DataReader.read_types_of_products(dataset_path)
        self.image_classes = DataReader.generate_images_classes_dict(
            self.types_of_products
        )

    def get_next_element(self):
        for label, products in self.image_classes.items():
            for product in products:
                yield label, product


# if __name__ == "__main__":
#     test_path = r"..\data\FruitRecognition\Test"
#     test_generator = TestProductsGenerator(test_path)
#     for l, p in test_generator.get_next_element():
#         print(l, p)
#         # print(l, p)

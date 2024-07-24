from utilities.data_reader import DataReader


class RepresentativesGenerator:
    def __init__(self, dataset_path):
        self.types_of_products = DataReader.read_types_of_products(dataset_path)
        self.image_classes = DataReader.generate_images_classes_dict(
            self.types_of_products
        )

    def get_next_element(self) -> tuple[int]:
        for i in range(0, len(self.types_of_products)):
            label = self.types_of_products[i]
            yield self.types_of_products[i], self.image_classes[label][0]

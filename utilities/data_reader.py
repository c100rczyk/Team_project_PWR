import os


class DataReader:
    def __init__(self):
        pass

    # The function below works correctly in 3 cases:
    # 1) images are only in subfolders
    # 2) images are only in subsubfolders
    # 3) images are both in subfolders and subsubfolders
    @staticmethod
    def read_types_of_products(dataset_path) -> list[str]:
        """
        Create list of all products. In this list will be paths to each product
        exists in dataset.
        Args:
            dataset_path: r"./data/VegFru/fru92_images/Training"
        Returns: list of paths (paths to all products)
            ['./data/VegFru/fru92_images/Training/guava', './data/VegFru/fru92_images/Training/hickory',
             './data/VegFru/fru92_images/Training/blood_orange', './data/VegFru/fru92_images/Training/prune', ...]
        """
        types = []
        subdirs_main = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
        # iterate over subdirectories in the main directory:
        for folder_name in subdirs_main:
            subdirs_main_path = os.path.join(dataset_path, folder_name)
            subdirs = [
                d
                for d in os.listdir(subdirs_main_path)
                if os.path.isdir(os.path.join(subdirs_main_path, d))
            ]
            # if there are no subsubdirectories, add the subdirectory name
            if len(subdirs) == 0:
                types.append(subdirs_main_path)
            else:
                for directory in subdirs:
                    types.append(os.path.join(subdirs_main_path, directory))
        return types

    @staticmethod
    def generate_images_classes_dict(products_types) -> dict[str, list[str]]:
        """
        Create dictionary contains paths to all images.
        Args:
            products_types: './data/VegFru/fru92_images/Training/guava'
        Returns:
            ['./data/VegFru/fru92_images/Training/guava/f_01_11_1066.jpg',
            './data/VegFru/fru92_images/Training/guava/f_01_11_0781.jpg',
            './data/VegFru/fru92_images/Training/guava/f_01_11_0250.jpg',
            './data/VegFru/fru92_images/Training/guava/f_01_11_0384.jpg', ... ]
            Etc... for all images of all products   len(all_products) == num of all images in dataset
        """
        all_products = dict()
        # Ścieżki do wszystkich zdjęć każdego z produktów
        for product in products_types:
            images_names = os.listdir(product)
            images_paths = [
                os.path.join(product, image_name) for image_name in images_names
            ]
            all_products[product] = images_paths

        return all_products

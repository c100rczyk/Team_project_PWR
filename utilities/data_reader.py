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
        all_products = dict()
        for product in products_types:
            images_names = os.listdir(product)
            images_paths = [
                os.path.join(product, image_name) for image_name in images_names
            ]
            all_products[product] = images_paths
        return all_products

import os
import shutil
import random
from utilities.data_reader import DataReader


def extract_dataset(source_path, training_path, validation_path, test_path):
    image_classes = DataReader.read_types_of_products(source_path)
    images_dict = DataReader.generate_images_classes_dict(image_classes)
    training_images = {}
    validation_images = {}
    test_images = {}
    training_coef = 0.7
    validation_coef = 0.15
    testing_coef = 0.15
    for label, images in images_dict.items():
        label = label.rsplit("\\", 1)[1]
        all_images_n = len(images)
        training_images_n = training_coef * all_images_n
        validation_images_n = validation_coef * all_images_n
        test_images_n = testing_coef * all_images_n
        print("Number of all images: " + str(all_images_n))
        print("Number of training images: " + str(training_images_n))
        print("Number of validation images: " + str(validation_images_n))
        print("Number of testing images: " + str(test_images_n))
        memory = set()
        training_images[label] = []
        while len(training_images[label]) < training_images_n - 1:
            img = random.choice(images)
            if img not in memory:
                memory.add(img)
                training_images[label].append(img)
        validation_images[label] = []
        while len(validation_images[label]) < validation_images_n - 1:
            img = random.choice(images)
            if img not in memory:
                memory.add(img)
                validation_images[label].append(img)
        test_images[label] = []
        while len(test_images[label]) < test_images_n - 1:
            img = random.choice(images)
            if img not in memory:
                memory.add(img)
                test_images[label].append(img)
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    for label, images in training_images.items():
        for img in images:
            path = os.path.join(training_path, label)
            if not os.path.exists(path):
                os.mkdir(path)
            file_path = os.path.join(path, img.rsplit("\\", 1)[1])
            shutil.copy(img, file_path)
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)
    for label, images in validation_images.items():
        for img in images:
            path = os.path.join(validation_path, label)
            if not os.path.exists(path):
                os.mkdir(path)
            file_path = os.path.join(path, img.rsplit("\\", 1)[1])
            shutil.copy(img, file_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for label, images in test_images.items():
        for img in images:
            path = os.path.join(test_path, label)
            if not os.path.exists(path):
                os.mkdir(path)
            file_path = os.path.join(path, img.rsplit("\\", 1)[1])
            shutil.copy(img, file_path)


if __name__ == "__main__":
    source_path = r""  # Source Dataset folder
    training_path = r"../data/FruitRecognition/Training"
    validation_path = r"../data/FruitRecognition/Validation"
    test_path = r"../data/FruitRecognition/Test"
    extract_dataset(
        source_path=source_path,
        training_path=training_path,
        validation_path=validation_path,
        test_path=test_path,
    )

import os.path
import tensorflow as tf
import numpy as np
from utilities.data_reader import DataReader
from utilities.mapping import Mapper
import os
from metrics.Product import Product
import app_utilities.config as config

image_size = 224, 224
margin = 0.5
representatives_path = r"../data/FruitRecognition/representatives/represent_5"


def load_representatives(path):
    """
    Create representatives dictionary with all necessary info
    Args:
        path: Get the images from this directory

    Returns:
        'representatives' dictionary. Key: 'label' , value: object of Product() class with all necessary attributes
    """
    products_paths = DataReader.read_types_of_products(path)
    products_dictionary = DataReader.generate_images_classes_dict(products_paths)
    representatives = {}
    mapper = Mapper(image_size)
    for label, products in products_dictionary.items():
        lab = label.rsplit("/", 1)[1]
        representatives[lab] = []
        for product in products:
            rep = Product()
            rep.label, rep.image = mapper.map_single_product(lab, product)
            rep.image_path = product
            representatives[lab].append(rep)
    for label, products in representatives.items():
        for product in products:
            product.image_to_predict = tf.reshape(product.image, shape=(1,) + image_size + (3,))
    return representatives

def add_representatives(image_to_add, label):
    """
    Creating new folder of "label" name if not exists.
    Create object of class Product() with all info about this image and save to Representatives.
    save new object to global representatives.
    save new image to new folder : 'label' representatives.

    Args:
        image_to_add: choosen image from camera capture / folder.
        label: name of object written in TextBox() on Gradio

    Returns:
        info about all done staff

    """
    rep_add = Product()
    mapper = Mapper(image_size)
    label_add, image_add = mapper.map_single_product(label=label, image_path=image_to_add)
    embedding_add = config.embedding_layer(tf.expand_dims(image_add, axis=0)).numpy()

    new_dir = f"{representatives_path}/{label_add}"
    #dodać sparwdzenie czy nie ma zdjęcia o już istniejącej nazwie
    image_name = f"{label_add}_{np.random.randint(low=0, high=100000, dtype=np.uint32)}.jpg"
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    rep_add.image = image_add
    rep_add.label = label_add
    rep_add.embedding = embedding_add
    rep_add.image_path = f"{new_dir}/{image_name}"
    rep_add.image_to_predict = tf.reshape(rep_add.image, shape=(1,) + image_size + (3,))

    if label_add not in config.representatives:
        config.representatives[label_add] = []

    config.representatives[label_add].append(rep_add)

    tf.keras.preprocessing.image.save_img(f"{new_dir}/{image_name}", rep_add.image)

    info = f"Added {label_add} to path {new_dir}. Name of image: {image_name}."
    return info
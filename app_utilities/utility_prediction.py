import tensorflow as tf
import numpy as np
from utilities.mapping import Mapper
from distance.EuclideanDistance import EuclideanDistance
from metrics.Metrics import Metrics
import utilities.config as config

image_size = 224, 224
margin = 0.5
representatives_path = r"../data/FruitRecognition/representatives/represent_5"


def predict(image):
    """
    Create embedding from captured image.
    Calculate mean distance between this image and each of representatives.
    Get top 5 most similar representatives.
    Args:
        image: captured from camera / get from folder

    Returns: top 5 labels and corresponding images
    """
    mapper = Mapper(image_size)
    label, image = mapper.map_single_product("", image)
    image = tf.reshape(image, shape=(1,) + image_size + (3,))
    if config.embedding_layer is None:
        raise ValueError("Embedding layer is not initialized! Please check your setup.")
    embedding = np.asarray(config.embedding_layer(image)).astype("float32")
    distances = np.zeros(len(config.representatives.keys()))
    labels = []
    j = 0
    for rep_label, products in config.representatives.items():
        sum = 0
        for product in products:
            sum += EuclideanDistance.calculate_distance((embedding, product.embedding))
        mean = sum / len(products)
        distances[j] = mean
        j += 1
        labels.append(rep_label)

    top5_dist, top5_labels = Metrics.find_top_5(distances, labels)
    predicted_images = []

    for label in top5_labels:
        img = tf.image.convert_image_dtype(
            config.representatives[label][0].image, dtype=tf.uint8
        )
        predicted_image = img.numpy()
        predicted_images.append(predicted_image)
        print(len(predicted_images))

    print(top5_dist, top5_labels)
    return (*top5_labels, *predicted_images)


def evaluate_representatives(embedding_layer, representatives):
    """
    Create embedding for all images in representatives
    Returns:
        representatives (class Product() ) with .embedding attribute
    """
    for label, products in representatives.items():
        for product in products:
            product.embedding = np.asarray(
                embedding_layer(product.image_to_predict)
            ).astype("float32")
    return representatives

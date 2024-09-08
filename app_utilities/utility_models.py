import tensorflow as tf
import keras
from keras import Model
from model.SiameseModel import SiameseModel
from distance.DistanceLayer import DistanceLayer
#from keras.src.applications.xception import Xception
from keras.api.keras.applications.vgg16 import VGG16
from keras import layers
from distance.EuclideanDistance import EuclideanDistance

image_size = 224, 224
margin = 0.5
representatives_path = r"../data/FruitRecognition/representatives/represent_5"


def create_triplet_model():
    """
        Create Triplet Model.
        Choose architecture and load previously trained weights.

        Returns: embedding_model - for create embedding from image
        """
    input_layer = keras.layers.Input(image_size + (3,))
    model_xception = VGG16(weights="imagenet", include_top=False)
    x = model_xception(input_layer)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Lambda(lambda param: tf.math.l2_normalize(param, axis=1))(x)
    embedding_model = keras.Model(input_layer, x)
    anchor_input = keras.layers.Input(name="anchor", shape=image_size + (3,))
    positive_input = keras.layers.Input(name="positive", shape=image_size + (3,))
    negative_input = keras.layers.Input(name="negative", shape=image_size + (3,))

    distances = DistanceLayer()(
        embedding_model(anchor_input),
        embedding_model(positive_input),
        embedding_model(negative_input),
    )

    siamese_model = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )
    siamese_network = SiameseModel(siamese_model, margin=margin)
    siamese_network.load_weights(
        r"../trained_models/VGG16_plus_dense_512_16_triplet.weights.h5"
    )
    return embedding_model, siamese_network

def create_contrastive_model():
    """
    Create Contrastive Model.
    Choose architecture and load previously trained weights.

    Returns: embedding_network - for create embedding from image
    """
    from tensorflow.keras.applications import VGG16
    modelVGG = VGG16(weights='imagenet', include_top=False)

    input_layer = keras.layers.Input((224, 224, 3))
    x = modelVGG(input_layer)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    embedding_network = keras.Model(input_layer, x)

    input_1 = keras.layers.Input(image_size + (3,))
    input_2 = keras.layers.Input(image_size + (3,))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer_distance = keras.layers.Lambda(EuclideanDistance(), output_shape=(1,))([tower_1, tower_2])
    siamese = keras.Model(inputs=[input_1, input_2], outputs=merge_layer_distance)

    for layer in modelVGG.layers:
        layer.trainable = False
    siamese.load_weights(r"../trained_models/VGG16_512_16_Contrastive.h5")

    return embedding_network, siamese
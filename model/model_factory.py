import keras
import tensorflow as tf

from CL_utilities import loss_function
from distance.DistanceLayer import DistanceLayer
from distance.EuclideanDistance import EuclideanDistance
from model.SiameseModel import SiameseModel
from keras import optimizers


class ModelFactory:
    @staticmethod
    def create_model(
        image_size, transfer_model, method, margin, optimizer, first_layer, second_layer
    ):
        input_layer = keras.layers.Input(image_size + (3,))
        model = transfer_model(input_layer)
        model = keras.layers.Flatten()(model)
        model = keras.layers.Dense(first_layer, activation="relu")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Dense(second_layer, activation="relu")(model)
        model = keras.layers.Lambda(lambda param: tf.math.l2_normalize(param, axis=1))(
            model
        )
        embedding_model = keras.Model(input_layer, model)
        embedding_model.name = "embedding"
        if method == "triplet_loss":
            anchor_input = keras.layers.Input(name="anchor", shape=image_size + (3,))
            positive_input = keras.layers.Input(
                name="positive", shape=image_size + (3,)
            )
            negative_input = keras.layers.Input(
                name="negative", shape=image_size + (3,)
            )

            distances = DistanceLayer()(
                embedding_model(anchor_input),
                embedding_model(positive_input),
                embedding_model(negative_input),
            )

            model = keras.Model(
                inputs=[anchor_input, positive_input, negative_input], outputs=distances
            )
            model = SiameseModel(model, margin=margin)
            model.compile(optimizer=optimizer, metrics=["accuracy"])
        elif method == "contrastive_loss":
            input_1 = keras.layers.Input(image_size + (3,))
            input_2 = keras.layers.Input(image_size + (3,))

            tower_1 = embedding_model(input_1)
            tower_2 = embedding_model(input_2)

            merge_layer_distance = keras.layers.Lambda(
                EuclideanDistance(), output_shape=(1,)
            )([tower_1, tower_2])
            model = keras.Model(inputs=[input_1, input_2], outputs=merge_layer_distance)
            model.compile(
                loss=loss_function.loss(margin=margin),
                optimizer=optimizer,
                metrics=["accuracy"],
            )

        return model

import keras
import tensorflow as tf

from CL_utilities import loss_function
from distance.DistanceLayer import DistanceLayer
from distance.EuclideanDistance import EuclideanDistance
from model.SiameseModel import SiameseModel


class ModelFactory:
    @staticmethod
    def create_model(
        image_size, transfer_model, method, margin, optimizer, *dense_layers_sizes
    ):
        """
        Create a tensorflow Model based on architecture type.
        Args:
            image_size: size of input images - tuple[uint8, uint8]
            transfer_model: pretrained model for transfer learning - keras.src.models.functional.Functional
            method: type of model architecture - str
            margin: margin for model learning - float
            optimizer: optimizer for model learning - keras.src.optimizers.Optimizer
            dense_layers_sizes: sizes of dense layers added to the base model - tuple[int, ...]
        Returns:
            embedding_layer - for evaluating embeddings of images - keras.src.models.model.Model
            model - prepared machine learning model - keras.src.models.model.Model
        """
        input_layer = keras.layers.Input(image_size + (3,))
        model = transfer_model(input_layer)
        model = keras.layers.Flatten()(model)
        for i in range(len(dense_layers_sizes)):
            model = keras.layers.Dense(dense_layers_sizes[i], activation="relu")(model)
            if i is not len(dense_layers_sizes) - 1:
                model = keras.layers.BatchNormalization()(model)
        # model = keras.layers.Dense(first_layer, activation="relu")(model)
        # model = keras.layers.BatchNormalization()(model)
        # model = keras.layers.Dense(second_layer, activation="relu")(model)
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

        return embedding_model, model

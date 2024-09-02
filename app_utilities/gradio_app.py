import gradio as gr
import keras
import tensorflow as tf
import numpy as np
from utilities.dataset_factory import DatasetFactory
from utilities.visualization import Visualizer
import keras
from keras import Model
from keras import optimizers
from model.SiameseModel import SiameseModel
from distance.DistanceLayer import DistanceLayer
from utilities.data_reader import DataReader
from utilities.mapping import Mapper
from keras.src.applications.xception import Xception

from distance.EuclideanDistance import EuclideanDistance
from metrics.Product import Product
from utilities.test_products_generator import TestProductsGenerator
from metrics.Metrics import Metrics
from livelossplot import PlotLossesKeras


image_size = 224, 224
margin = 0.5
representatives_path = r"..\data\FruitRecognition\representatives\represent_5"


def load_representatives(path):
    products_paths = DataReader.read_types_of_products(path)
    products_dictionary = DataReader.generate_images_classes_dict(products_paths)
    representatives = {}
    mapper = Mapper(image_size)
    for label, products in products_dictionary.items():
        lab = label.rsplit("\\", 1)[1]
        representatives[lab] = []
        for product in products:
            rep = Product()
            rep.label, rep.image = mapper.map_single_product(lab, product)
            rep.image_path = product
            representatives[lab].append(rep)
    for label, products in representatives.items():
        for product in products:
            product.image = tf.reshape(product.image, shape=(1,) + image_size + (3,))
    return representatives


def evaluate_representatives(embedding_layer, representatives):
    for label, products in representatives.items():
        for product in products:
            product.embedding = np.asarray(embedding_layer(product.image)).astype(
                "float32"
            )
    return representatives


def create_model():
    input_layer = keras.layers.Input(image_size + (3,))
    model_xception = Xception(weights="imagenet", include_top=False)
    x = model_xception(input_layer)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
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
        r"../trained_models/Xception_VegFru_dense_512_256_triplet.weights.h5"
    )
    return embedding_model, siamese_network


def predict(image):
    mapper = Mapper(image_size)
    label, image = mapper.map_single_product("", image)
    image = tf.reshape(image, shape=(1,) + image_size + (3,))
    embedding = np.asarray(embedding_layer(image)).astype("float32")
    distances = np.zeros(len(representatives.keys()))
    labels = []
    j = 0
    for rep_label, products in representatives.items():
        sum = 0
        for product in products:
            sum += EuclideanDistance.calculate_distance((embedding, product.embedding))
        mean = sum / len(products)
        distances[j] = mean
        j += 1
        labels.append(rep_label)

    top5_dist, top5_labels = Metrics.find_top_5(distances, labels)
    predicted_images = []
    # for label in top5_labels:
    #     img = tf.image.convert_image_dtype(
    #         representatives[label][0].image, dtype=tf.uint8
    #     )
    #     predicted_images.append(img.numpy())
    print(top5_dist, top5_labels)
    # return predicted_images
    return top5_labels


embedding_layer, model = create_model()
representatives = load_representatives(representatives_path)
representatives = evaluate_representatives(embedding_layer, representatives)

with gr.Blocks() as app:
    # app.load(fn=create_model, outputs=[embedding_layer, model])
    # app.load(fn=load_representatives(representatives_path), outputs=[representatives])
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image()
            predict_btn = gr.Button(value="Predict")
        with gr.Column(scale=2):
            with gr.Row():
                # predictions = [
                #     gr.Image(type="numpy", interactive=False, height=224, width=224),
                #     gr.Image(type="numpy", interactive=False, height=224, width=224),
                #     gr.Image(type="numpy", interactive=False, height=224, width=224),
                #     gr.Image(type="numpy", interactive=False, height=224, width=224),
                #     gr.Image(type="numpy", interactive=False, height=224, width=224),
                # ]
                predictions = [
                    gr.Textbox(),
                    gr.Textbox(),
                    gr.Textbox(),
                    gr.Textbox(),
                    gr.Textbox(),
                ]
    predict_btn.click(predict, inputs=[image], outputs=predictions)


if __name__ == "__main__":
    app.launch(share=True)

import os.path
from pathlib import Path
import cv2
import gradio as gr
import tensorflow as tf
import numpy as np
import keras
from keras.src.applications.vgg16 import VGG16
from utilities.data_reader import DataReader
from utilities.mapping import Mapper
from utilities.config import ConfigReader, Config
from model.model_factory import ModelFactory
from distance.EuclideanDistance import EuclideanDistance
from metrics.Product import Product
from metrics.Metrics import Metrics

config_reader = ConfigReader("config.json")
config = Config(config_reader.load_config())
image_size = (
    config.image_properties.image_height,
    config.image_properties.image_width,
)


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
            product.image_to_predict = tf.reshape(
                product.image, shape=(1,) + image_size + (3,)
            )
    return representatives


def add_representatives(images_to_add, label):
    """
    Creating new folder of "label" name if not exists.
    Create object of class Product() with all info about this image and save to Representatives.
    save new object to global representatives.
    save new image to new folder : 'label' representatives.

    Args:
        images_to_add: choosen images from camera capture / folder.
        label: name of object written in TextBox() on Gradio

    Returns:
        info about all done staff

    """
    if not label:
        gr.Warning("Please provide a label for your product")
        return
    mapper = Mapper(image_size)
    mapped_images = []
    for image_path in images_to_add:
        label, image = mapper.map_single_product(label, image_path)
        mapped_images.append(
            Product(
                label,
                os.path.join(
                    config.paths.representatives,
                    *[
                        label,
                        f"{label}_{np.random.randint(low=0, high=100000, dtype=np.uint32)}",
                    ],
                ),
                image,
                embedding_layer(tf.expand_dims(image, axis=0)).numpy(),
            )
        )

    new_dir = os.path.join(config.paths.representatives, label)
    # dodać sparwdzenie czy nie ma zdjęcia o już istniejącej nazwie
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    if label not in representatives:
        representatives[label] = []

    for image in mapped_images:
        representatives[label].append(image)
        tf.keras.preprocessing.image.save_img(image.image_path + ".jpg", image.image)
        image.image_to_predict = tf.reshape(image.image, shape=(1,) + image_size + (3,))
    info = f"Added {label} to path {new_dir}"
    return gr.Textbox(info, label="Operation result", visible=True, interactive=False)


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


def predict(image):
    """
    Create embedding from captured image.
    Calculate mean distance between this image and each of representatives.
    Get top 5 most similar representatives.
    Args:
        image: captured from camera / get from folder

    Returns: top 5 labels and corresponding images
    """
    labels_gradio = [
        gr.Textbox(
            "", container=False, show_label=False, interactive=False, visible=False
        ),
        gr.Textbox(
            "", container=False, show_label=False, interactive=False, visible=False
        ),
        gr.Textbox(
            "", container=False, show_label=False, interactive=False, visible=False
        ),
        gr.Textbox(
            "", container=False, show_label=False, interactive=False, visible=False
        ),
        gr.Textbox(
            "", container=False, show_label=False, interactive=False, visible=False
        ),
    ]
    images_gradio = [
        gr.Image(
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=False,
            show_label=False,
        ),
        gr.Image(
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=False,
            show_label=False,
        ),
        gr.Image(
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=False,
            show_label=False,
        ),
        gr.Image(
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=False,
            show_label=False,
        ),
        gr.Image(
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=False,
            show_label=False,
        ),
    ]
    if not np.any(image):
        gr.Warning("Please upload an image for classification.")
        return *labels_gradio, *images_gradio
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

    for label in top5_labels:
        img = tf.image.convert_image_dtype(
            representatives[label][0].image, dtype=tf.uint8
        )
        predicted_image = img.numpy()
        predicted_images.append(predicted_image)
        print(len(predicted_images))

    print(top5_dist, top5_labels)
    # Nie przypisuje wartości nie wiedzieć czemu
    # for i in range(len(top5_dist)):
    #     labels_gradio[i].value = top5_labels[i]
    #     labels_gradio[i].visible = True
    # for i in range(len(predicted_images)):
    #     images_gradio[i].value = predicted_images[i]
    #     images_gradio[i].visible = True
    return *[
        gr.Textbox(
            value=top5_labels[0],
            container=False,
            show_label=False,
            interactive=False,
            visible=True,
        ),
        gr.Textbox(
            value=top5_labels[1],
            container=False,
            show_label=False,
            interactive=False,
            visible=True,
        ),
        gr.Textbox(
            value=top5_labels[2],
            container=False,
            show_label=False,
            interactive=False,
            visible=True,
        ),
        gr.Textbox(
            value=top5_labels[3],
            container=False,
            show_label=False,
            interactive=False,
            visible=True,
        ),
        gr.Textbox(
            value=top5_labels[4],
            container=False,
            show_label=False,
            interactive=False,
            visible=True,
        ),
    ], *[
        gr.Image(
            value=predicted_images[0],
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=True,
            show_label=False,
        ),
        gr.Image(
            value=predicted_images[1],
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=True,
            show_label=False,
        ),
        gr.Image(
            value=predicted_images[2],
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=True,
            show_label=False,
        ),
        gr.Image(
            value=predicted_images[3],
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=True,
            show_label=False,
        ),
        gr.Image(
            value=predicted_images[4],
            type="numpy",
            interactive=False,
            height=224,
            width=224,
            visible=True,
            show_label=False,
        ),
    ]


embedding_layer, model = ModelFactory.create_model(
    image_size,
    VGG16(weights="imagenet", include_top=False),
    "contrastive_loss",
    config.margin,
    keras.optimizers.Adam(0.001),
    *(512, 16),
)
model.load_weights(
    config.paths.model_weights.VGG16_FruitRecognition_dense_512_16_contrastive
)
representatives = load_representatives(config.paths.representatives)
representatives = evaluate_representatives(embedding_layer, representatives)


with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown(
            """
            # Welcome to the Fruit Classifier
            ## Image Classification
            Below you can upload an image of a fruit for the model to recognize.
            You can either upload a file from a drive or capture a new image using a camera.
            """
        )
    with gr.Row():
        with gr.Column(scale=1):
            # load_img_btn = gr.Button("Load Image")

            image_to_predict = gr.Image(label="Loaded Image")
            predict_btn = gr.Button(value="Predict")

        with gr.Column(scale=2):
            with gr.Row():
                predictions_images = [
                    gr.Image(visible=False),
                    gr.Image(visible=False),
                    gr.Image(visible=False),
                    gr.Image(visible=False),
                    gr.Image(visible=False),
                ]
                prediction_labels = [
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                    gr.Textbox(visible=False),
                ]

    with gr.Row():
        gr.Markdown(
            """
            ## New Product
            Below you can upload images representing a new product class
            """
        )

    with gr.Row():
        with gr.Column():
            new_product_representatives = gr.UploadButton(
                "Upload images", file_count="multiple", file_types=["image"]
            )
            label_of_object = gr.Textbox(label="Enter the label for the new product")
        with gr.Column():
            # add_new_object = gr.Button(value="Add New Object to shop")
            output_info = gr.Textbox(visible=False)

    # load_img_btn.click(fn=capture_and_load_image, inputs=[], outputs=[image_to_predict])
    predict_btn.click(
        predict,
        inputs=[image_to_predict],
        outputs=prediction_labels + predictions_images,
    )
    new_product_representatives.upload(
        add_representatives,
        inputs=[new_product_representatives, label_of_object],
        outputs=[output_info],
    )


if __name__ == "__main__":
    app.launch(share=True)

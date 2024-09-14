import gradio as gr
from app_utilities.utility_models import create_contrastive_model
from app_utilities.utility_representatives import (
    load_representatives,
    add_representatives,
)
from app_utilities.utility_prediction import evaluate_representatives, predict
from app_utilities.utility_camera import capture_and_load_image
from utilities import config

representatives_path = r"../data/FruitRecognition/representatives/represent_5"

embedding_layer, model = create_contrastive_model()
representatives = load_representatives(representatives_path)
representatives = evaluate_representatives(embedding_layer, representatives)

config.representatives = representatives
config.embedding_layer = embedding_layer

with gr.Blocks() as app:

    with gr.Row():
        with gr.Column(scale=1):
            load_img_btn = gr.Button("Load Image")
            image_to_predict = gr.Image(label="Loaded Image")
            load_img_btn.click(
                fn=capture_and_load_image, inputs=[], outputs=[image_to_predict]
            )

            predict_btn = gr.Button(value="Predict")
        with gr.Column(scale=2):
            with gr.Row():
                predictions_images = [
                    gr.Image(type="numpy", interactive=False, height=224, width=224),
                    gr.Image(type="numpy", interactive=False, height=224, width=224),
                    gr.Image(type="numpy", interactive=False, height=224, width=224),
                    gr.Image(type="numpy", interactive=False, height=224, width=224),
                    gr.Image(type="numpy", interactive=False, height=224, width=224),
                ]
                predictions_labels = [
                    gr.Textbox(),
                    gr.Textbox(),
                    gr.Textbox(),
                    gr.Textbox(),
                    gr.Textbox(),
                ]
    predict_btn.click(
        predict,
        inputs=[image_to_predict],
        outputs=predictions_labels + predictions_images,
    )

    ## Dodawanie do reprezentantów nowych obiektów.
    with gr.Row():
        add_new_object = gr.Button(value="Add New Object to shop")
        label_of_object = gr.Textbox(label="Enter the label")
        output_info = gr.Textbox()
        add_new_object.click(
            fn=add_representatives,
            inputs=[image_to_predict, label_of_object],
            outputs=[output_info],
        )


if __name__ == "__main__":
    app.launch(share=True)

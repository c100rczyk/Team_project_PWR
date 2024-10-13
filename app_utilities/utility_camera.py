import cv2
import tensorflow as tf
import os
from app_utilities.app_config import image_size
import gradio as gr


def capture_and_load_image(camera_id=0):
    """
    Capture image from camera and convert this image appropriate
    Args:
        camera_id: id of camera (int)
            run command: ls dev/video*     there is all accessible video devices from which we cen get image
            There can be 2 objects from one camera. One is form image flow (we want this one)
            second for technical configuration.

    Returns:
        image in numpy format
    """
    camera = cv2.VideoCapture(camera_id)
    return_value, image = camera.read()
    output_path = "app_utilities/przechwycone_obrazy"
    image_name = f"zdj_{camera_id}.png"
    image_path = os.path.join(output_path, image_name)
    if not return_value:
        gr.Warning("Could not capture an image.")
        camera.release()
        cv2.destroyAllWindows()
        return gr.Image(label="Loaded Image", sources=["upload"])
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not cv2.imwrite(image_path, image):
        print("Could not write an image.")
        camera.release()
        cv2.destroyAllWindows()
        return gr.Image(label="Loaded Image", sources=["upload"])
    camera.release()
    cv2.destroyAllWindows()

    captured_image = tf.io.read_file(image_path)
    captured_image = tf.image.decode_jpeg(captured_image, channels=3)
    captured_image = tf.image.convert_image_dtype(captured_image, dtype=tf.float32)
    captured_image = tf.image.resize(captured_image, image_size)
    captured_image_np = captured_image.numpy()
    return gr.Image(value=captured_image_np, label="Loaded Image", sources=["upload"])

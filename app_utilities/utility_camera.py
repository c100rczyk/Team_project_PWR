import cv2
import tensorflow as tf
import numpy as np
import os
from gradio_app import image_size


def capture_and_load_image(object=0):
    """
    Capture image from camera and convert this image appropriate
    Args:
        object: id of camera (int)
            run command: ls dev/video*     there is all accessible video devices from which we cen get image
            There can be 2 objects from one camera. One is form image flow (we want this one)
            second for technical configuration.

    Returns:
        image in numpy format
    """
    camera = cv2.VideoCapture(object)
    return_value, image = camera.read()
    cv2.waitKey()
    output_path = "app_utilities/przechwycone_obrazy"
    image_name = f"zdj_{id}.png"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not cv2.imwrite(os.path.join(output_path, image_name), image):
        return
    print("written")
    camera.release()
    cv2.destroyAllWindows()

    captured_image = tf.io.read_file(os.path.join(output_path, image_name))
    captured_image = tf.image.decode_jpeg(captured_image, channels=3)
    captured_image = tf.image.convert_image_dtype(captured_image, dtype=tf.float32)
    captured_image = tf.image.resize(captured_image, image_size)
    captured_image_np = captured_image.numpy()
    return captured_image_np

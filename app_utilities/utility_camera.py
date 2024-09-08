import cv2
import tensorflow as tf
import numpy as np
image_size = 224, 224
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
    output_path = f"przechwycone_obrazy/zdj_{id}.png"
    cv2.imwrite(output_path, image)
    camera.release()
    cv2.destroyAllWindows()

    if type(output_path) is np.ndarray:
        captured_image = tf.convert_to_tensor(output_path)
    else:
        captured_image = tf.io.read_file(output_path)
        captured_image = tf.image.decode_jpeg(captured_image, channels=3)
    captured_image = tf.image.convert_image_dtype(captured_image, dtype=tf.float32)
    captured_image = tf.image.resize(captured_image, image_size)
    captured_image_np = captured_image.numpy()
    return captured_image_np
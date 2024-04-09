
import tensorflow as tf

class Generate_pack_of_images:
    # IMAGE_SIZE = (258, 320)
    # batch_size = 8
    def __init__(self, image_size=(258,320), batch_size=8):
        self.image_size = image_size
        self.batch_size = batch_size

    def decode_and_resize(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, self.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def __call__(self, image_path1, image_path2, value):
        # return tf.py_function(decode_and_resize, [image_path], tf.float32)
        img1 = self.decode_and_resize(image_path1)
        img2 = self.decode_and_resize(image_path2)
        return ((img1, img2), value)


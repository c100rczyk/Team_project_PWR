import tensorflow as tf


class MapFunction:
    def __init__(self, imageSize):
        # define the image width and height
        self.imageSize = imageSize

    def decode_and_resize(self, imagePath):
        # read and decode the image path
        image = tf.io.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)
        # convert the image data type from uint8 to float32 and then resize
        # the image to the set image size
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, self.imageSize)
        # return the image
        return image

    def __call__(self, anchor, positive, negative):
        anchor = self.decode_and_resize(anchor)
        positive = self.decode_and_resize(positive)
        negative = self.decode_and_resize(negative)
        # return the anchor, positive and negative processed images
        return anchor, positive, negative

import numpy
import tensorflow as tf


class Mapper:
    """
    Get paths of pairs/triplets
    Return images of paths that has expected sizes and types
    """

    def __init__(self, image_size):
        # define the image width and height
        self.imageSize = image_size

    def _decode_and_resize(self, image_path):
        # read and decode the image path
        if type(image_path) is numpy.ndarray:
            image = tf.convert_to_tensor(image_path)
        else:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
        # convert the image data type from uint8 to float32 and then resize
        # the image to the set image size
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, self.imageSize)
        return image

    def _map_triplet_loss(self, anchor, positive, negative):
        anchor = self._decode_and_resize(anchor)
        positive = self._decode_and_resize(positive)
        negative = self._decode_and_resize(negative)
        return anchor, positive, negative

    def _map_contrastive_loss(self, image_path1, image_path2, label):
        img1 = self._decode_and_resize(image_path1)
        img2 = self._decode_and_resize(image_path2)
        return (img1, img2), label

    def map_single_product(self, label, image_path):
        return label, self._decode_and_resize(image_path)

    def __call__(self, *args, method: str):
        """
        :param args: tuple(tensor, tensor, tensor) | tuple(tensor, tensor, label)
        :param method: str"triplet_loss" | str"contrastive_loss"
        """
        if method == "triplet_loss":
            return self._map_triplet_loss(*args)
        elif method == "contrastive_loss":
            return self._map_contrastive_loss(*args)
        elif method == "representatives":
            return self.map(*args)

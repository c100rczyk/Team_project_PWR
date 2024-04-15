import tensorflow as tf
import keras


class EuclideanDistance:
    """
    Znalezienie odległości euklidesowej pomiędzy dwoma wektorami:

    Arguments:
        vects: Lista zawierające dwa tensory tej samej długości

    Returns:
        Tensor containing euclidean distance pomiędzy podanymi wektorami
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, vects, mask=None):
        x, y = vects

        sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_square, keras.backend.epsilon()))

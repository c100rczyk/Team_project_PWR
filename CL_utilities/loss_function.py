import tensorflow as tf

"""
    Source: https://keras.io/examples/vision/siamese_contrastive/
"""


def loss(margin=1):
    """
    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
        shoud be classified as dissimilar "1"
    Returns:
        tensor with contrastive loss as floating point value
    """

    # 0-same ,  1-different

    # contrastive_loss = (1-y_true)*

    def contrastive_loss(y_true, y_pred):
        """Calculate the contrastive loss
        Arguments:
            y_true: List of labels, each label is of type "float32"
            y_pred: List of predictions
            y_pred to przewidywane odległości między parami danych, które model stara się nauczyć
        Returns:
            A tensor containing contrastive loss value (folat)
        """
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - (y_pred), 0))
        return tf.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss

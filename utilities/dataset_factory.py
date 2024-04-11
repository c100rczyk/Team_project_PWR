import tensorflow as tf
from utilities.triplet_generator import TripletGenerator
from utilities.mapping import MapFunction


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def build_dataset(
        ds_path, image_size, batch_size, max_iterations, output_signature
    ):
        generator = TripletGenerator(ds_path, max_iterations)
        return (
            tf.data.Dataset.from_generator(
                generator.get_next_element, output_signature=output_signature
            )
            .map(
                lambda anchor, positive, negative: (
                    MapFunction(image_size)(anchor, positive, negative)
                )
            )
            .batch(batch_size)
        )

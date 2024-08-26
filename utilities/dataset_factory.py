import tensorflow as tf
from utilities.triplet_generator import TripletGenerator
from utilities.contrastive_generator import ContrastiveGenerator
from utilities.mapping import Mapper
from utilities.output_signatures import OutputSignature


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def build_dataset(ds_path, image_size, batch_size, max_iterations, method):
        """
        Build Tensors with prepared pairs of images and labels. Ready to train in model.
        Args:
            ds_path: get images from
            image_size: compute image to this size (specific for model: ResNet, VGG,...)
            batch_size: how many pairs of images to process at one time
            max_iterations:
            method: contrastive / triplet

        Returns:
            <_BatchDataset element_spec=(
            (TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None),
            TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None)),
            TensorSpec(shape=(None,), dtype=tf.float32, name=None)
            )>
        """
        generator = None
        output_signature = None
        if method == "triplet_loss":
            generator = TripletGenerator(ds_path, max_iterations)
            output_signature = OutputSignature.triplet_loss
        elif method == "contrastive_loss":
            generator = ContrastiveGenerator(ds_path, max_iterations)
            output_signature = OutputSignature.contrastive_loss
        return (
            tf.data.Dataset.from_generator(
                generator.get_next_element, output_signature=output_signature
            )
            .map(
                lambda anchor, positive, negative: (
                    Mapper(image_size)(anchor, positive, negative, method=method)
                )
            )
            .batch(batch_size)
        )

    def give_paths_to_test(ds_path, max_iterations):
        contr = ContrastiveGenerator(ds_path, max_iterations)
        return contr.dict_all_images_name_path

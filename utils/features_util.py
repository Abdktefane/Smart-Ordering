import os
import errno
import argparse
import numpy as np
import cv2
from utils.hyper_params import default_params
import tensorflow.compat.v1 as tf


def create_box_encoder(
        model_filename=default_params['feature_model_path'],
        input_name="images",
        output_name="features",
        batch_size=default_params['max_boxes_to_draw']
):
    """
    @param model_filename: the path to frozen graph
    @return: encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    """
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape where ==>  height = patch_shape[0], width = patch_shape[1]
        target_width = patch_shape[1]
        target_height = patch_shape[0]
        target_ratio = target_width / target_height
        bbox_height = bbox[3] - bbox[1]
        # here we extract new width from equation (width/height) * height => width
        new_width = target_ratio * bbox_height
        # modify x_min with new width
        bbox[0] -= new_width / 2

    bbox = bbox.astype(np.int)
    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images", output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        # input_var = Tensor("images:0", shape=(None, 128, 64, 3), dtype=uint8) where None is batch size
        self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name("%s:0" % input_name)
        # output_var = Tensor("features:0", shape=(None, 128), dtype=float32) where None is batch size
        self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name("%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        # self.feature_dim = 128
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        # self.image_shape = [128, 64, 3]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x},
            out,
            batch_size
        )
        return out

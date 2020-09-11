from abc import ABC, abstractmethod


class Detector(ABC):
    """
    This class represents a blueprint for other detectors models and should adopt it.

    Parameters
    ----------
    config : Config
        base attribute for detectors

    Attributes
    ----------

    """

    def __init__(self, config):
        self.config= config

    def detect(self, images):
        """
        Serve a list of image arrays.

        Args:
          images: A list of image content with each image has shape [height, width, 3] and u_int_8 type.

        Returns:
          A list of Detection.
        """
        pass

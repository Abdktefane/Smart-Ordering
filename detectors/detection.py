# vim: expandtab:ts=4:sw=4
import numpy as np
from utils import util


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    detection : array_like
        `(image_id, y_min, x_min, y_max, x_max, score, class)`.
    feature: bool | NoneType
    A feature vector that describes the object contained in this image.


    Attributes
    ----------
    base : ndarray
        Bounding box in format `(y_min, x_min, y_max, x_max, score, class )`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, detection, feature=None):
        self.base = np.array(detection[1:5])
        self.confidence = detection[5]
        self.class_ = detection[6]
        if feature is not None:
            self.feature = np.asarray(feature, dtype=np.float32)
        else:
            self.feature = None

    def set_feature(self, feature):
        # self.feature = feature
        self.feature = np.asarray(feature, dtype=np.float32)

    def get_feature(self):
        return self.feature

    def to_base(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        return self.base.copy()

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        return np.array((self.base[1], self.base[0], self.base[3], self.base[2]))

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,height)`,
         where the aspect ratio is `width / height`.
        """
        return util.box_to_coco(self.base.copy())

    def as_np(self):
        return np.array((self.base.copy(), self.confidence, self._class))

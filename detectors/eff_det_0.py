from detectors.detector import Detector
from detectors.detection import Detection
from utils.util import parse_image_size
import inference
from detectors.detection import Detection
import hparams_config
from utils.hyper_params import default_params


class EfficientDet0(Detector):
    """
    This class represents a blueprint for other detectors models and should adopt it.

    Parameters
    ----------
    config : Config
        base attribute for detectors

    image_size: string or int
        the input shape of image "input size of model"

    min_score_thresh: float
        minimal score threshold for filtering predictions.

    max_boxes_to_draw: int
        the maximum number of boxes per image.

    Attributes
    ----------

    """

    def __init__(
            self,
            config=hparams_config.get_efficientdet_config('efficientdet-d0'),
            image_size=default_params['image_size'],
            min_score_thresh=default_params['min_score_thresh'],
            max_boxes_to_draw=default_params['max_boxes_to_draw']
    ):
        super(EfficientDet0, self).__init__(config)
        config.is_training_bn = False
        config.image_size = parse_image_size(image_size)
        config.nms_configs.score_thresh = min_score_thresh
        config.nms_configs.max_output_size = max_boxes_to_draw
        config.anchor_scale = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.config = config
        self.driver = inference.ServingDriver(
            model_name="efficientdet-d0",
            ckpt_path="efficientdet-d0",
            batch_size=1,
            min_score_thresh=min_score_thresh,
            max_boxes_to_draw=max_boxes_to_draw,
            use_xla=default_params['use_xla'],
            model_params=config.as_dict()
        )
        saved_model_dir = 'savedmodeldir'
        self.driver.load(saved_model_dir)

    def __call__(self, image):
        return self._detect(image)

    def _detect(self, image):
        """

        Args:
          image: image with shape [height, width, 3] and u_int_8 type.

        Returns:
          A list of filtered Detection.
        """
        """
        height, width = parse_image_size(self.config.image_size)
        np.resize(image, (height, width))
        image = np.array(image)
        """
        thresh = self.config.nms_configs.score_thresh
        detections = self.driver.serve_images([image])
        return [Detection(detection) for detection in detections[0] if detection[5] > thresh and detection[6] == 1]

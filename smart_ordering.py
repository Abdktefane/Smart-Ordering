import tensorflow as tf
import cv2
import datetime
import threading
import argparse
from detectors.eff_det_0 import EfficientDet0
from trackers.deep_tracker import DeepSort
from trackers.sort_tracker import Sort
from utils.overlay_util import paint_overlay
from utils.hyper_params import default_params


class SmartOrdering(object):
    def __init__(
            self,
            detector=default_params["detector_model_name"],
            tracker=default_params["tracker_model_name"]
    ):
        if detector == "efficientdet-d0":
            self.detector = EfficientDet0()
        else:
            raise ValueError("{} unsupported as detectors type".format(detector))

        if tracker == "sort":
            self.tracker = Sort()
        elif tracker == "deep_sort":
            self.tracker = DeepSort()
        else:
            raise ValueError("{} unsupported as tracker type".format(tracker))

    def __call__(self, frame, fps_print: bool = True):
        self._frame_processing(frame, fps_print)

    def _frame_processing(self, frame, fps_print: bool = True):
        starting_time = datetime.datetime.now()
        detections = self.detector(frame)
        if len(detections) == 0:
            threading.Thread(cv2.imshow("Image", frame)).start()
            # TODO we should steel run tracker with empty detection
            return
        tracks = self.tracker(detections, frame)
        fps = None
        if fps_print:
            fps = 1000 / ((datetime.datetime.now() - starting_time).total_seconds() * 1000)
        threading.Thread(self._visualize_image(
            frame=frame,
            trackers=tracks,
            fps=fps
        )).start()

    def _visualize_image(self, frame, trackers=None, fps=None):
        frame = paint_overlay(
            image=frame,
            prediction=trackers,
            min_score_thresh=self.detector.config.nms_configs.score_thresh,
            max_boxes_to_draw=self.detector.config.nms_configs.max_output_size,
            line_thickness=default_params['line_thickness'],
            fps=fps
        )
        cv2.imshow("Image", frame)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="smart ordering settings")
    parser.add_argument(
        "--max_boxes_to_draw",
        default=default_params["max_boxes_to_draw"],
        help="the max tracking object"
    )

    parser.add_argument(
        "--image_size",
        default=default_params["image_size"],
        help="size of input image"
    )

    parser.add_argument(
        "--min_score_thresh",
        default=default_params["min_score_thresh"],
        help="the minimum threshold for accepting detection"
    )

    parser.add_argument(
        "--line_thickness",
        default=default_params["line_thickness"],
        help="line thickness for draw bounding box"
    )

    parser.add_argument(
        "--max_boxes_to_draw",
        default=default_params["max_boxes_to_draw"],
        help="the max tracking object"
    )

    parser.add_argument(
        "--max_cosine_distance",
        default=default_params["max_cosine_distance"],
        help="threshold for cosine similarity"
    )
    parser.add_argument(
        "--nn_budget",
        default=default_params["nn_budget"],
        help="size of tracks gallery"
    )

    parser.add_argument(
        "--max_age",
        default=default_params["max_age"],
        help="age of tracks"
    )

    parser.add_argument(
        "--nn_init",
        default=default_params["nn_init"],
        help="tentative time of tracks"
    )

    parser.add_argument(
        "--detector_model_name",
        default=default_params["detector_model_name"],
        help="detector model type"
    )

    parser.add_argument(
        "--detector_saved_model_dir",
        default=default_params["detector_saved_model_dir"],
        help="path to detector model graph file"
    )
    parser.add_argument(
        "--eff_det_0_model_path",
        default=default_params["eff_det_0_model_path"],
        help="path to efficient det 0 model checkpoint"
    )
    parser.add_argument(
        "--tracker_model_name",
        default=default_params["tracker_model_name"],
        help="tracker model type"
    )

    parser.add_argument(
        "--feature_model_path",
        default=default_params["feature_model_path"],
        help="path to feature extractor model checkpoint"
    )
    args = parser.parse_args()
    default_params["max_boxes_to_draw"] = args.max_boxes_to_draw
    default_params["image_size"] = args.image_size
    default_params["min_score_thresh"] = args.min_score_thresh
    default_params["line_thickness"] = args.line_thickness
    default_params["max_cosine_distance"] = args.max_cosine_distance
    default_params["nn_budget"] = args.nn_budget
    default_params["max_age"] = args.max_age
    default_params["nn_init"] = args.nn_init
    default_params["detector_model_name"] = args.detector_model_name
    default_params["detector_saved_model_dir"] = args.detector_saved_model_dir
    default_params["eff_det_0_model_path"] = args.eff_det_0_model_path
    default_params["tracker_model_name"] = args.tracker_model_name
    default_params["feature_model_path"] = args.feature_model_path


def main():
    parse_args()
    # Use 'mixed_float16' if running on GPUs. or 'float32' on CPU
    policy = tf.keras.mixed_precision.experimental.Policy('float32')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    smart_ordering = SmartOrdering()
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        smart_ordering(frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


if __name__ == '__main__':
    main()

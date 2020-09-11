import tensorflow as tf
import cv2
import datetime
import threading
from detectors.eff_det_0 import EfficientDet0
from trackers.deep_tracker import DeepSort
from trackers.sort_tracker import Sort
from utils.overlay_util import paint_overlay
from utils.hyper_params import default_params


class SmartOrdering(object):
    def __init__(self, detector="eff_det_0", tracker="sort"):
        if detector == "eff_det_0":
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
            # TODO we steel should run tracker with empty detection
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


def main():
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

import tensorflow as tf
import numpy as np
import cv2
import hparams_config
import utils
import inference
from visualize_util import overlay_util
import datetime

from deep_sort import generate_detections as feature_util
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from deep_sort.detection import Detection

model_name = 'efficientdet-d0'
image_size = '512x512'
batch_size = 1
use_xla = False
nms_score_thresh = 0.3
detection_threshold = 0.3
line_thickness = 4
nms_max_output_size = 20
ckpt = "efficientdet-d0"
saved_model_dir = "savedmodeldir"
config = hparams_config.get_efficientdet_config('efficientdet-d0')
feature_extractor = feature_util.create_box_encoder(model_filename='networks/mars-small128.pb')
max_cosine_distance = 0.2
nn_budget = 100
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
deep_sort = Tracker(metric)


def build_model():
    config.is_training_bn = False
    config.image_size = utils.parse_image_size(image_size)
    config.nms_configs.score_thresh = nms_score_thresh
    config.nms_configs.max_output_size = nms_max_output_size
    config.anchor_scale = [1.0, 1.0, 1.0, 1.0, 1.0]
    driver = inference.ServingDriver(
        model_name,
        ckpt,
        batch_size,
        min_score_thresh=nms_score_thresh,
        max_boxes_to_draw=nms_max_output_size,
        use_xla=use_xla,
        model_params=config.as_dict()
    )
    driver.load(saved_model_dir)
    return driver


def process_image(frame, driver, fps_print: bool = False):
    starting_time = datetime.datetime.now()
    height, width = utils.parse_image_size(config.image_size)
    np.resize(frame, (height, width))
    frame = np.array(frame)
    detections = driver.serve_images([frame])
    # TODO filter detections class here and max detections here
    filtered_detection = [Detection(detection) for detection in detections[0] if
                          detection[5] > detection_threshold and detection[6] == 1]
    if len(filtered_detection) == 0:
        # TODO draw empty image then return
        return
    featured_detection = feature_extractor(frame,
                                           [d.to_tlbr() for d in filtered_detection])  # TODO chek if she want to toXSAH
    [det.set_feature(feature) for det, feature in zip(filtered_detection, featured_detection)]
    trackers = track(filtered_detection)
    fps = None
    '''
        if fps_print:
        elapsed_time = datetime.datetime.now() - starting_time
        fps = 1000 / (elapsed_time.total_seconds() * 1000)
        # print("inference time: {}, FPS: {} ".format(elapsed_time.total_seconds() * 1000, fps))
    # threading.Thread(visualize_image(driver, frame, detections, fps)).start()
    threading.Thread(visualize_image(frame, trackers, fps)).start()
    '''


def visualize_image(frame, trackers, fps=None):
    frame = overlay_util.paint_overlay(frame, trackers, detection_threshold, nms_max_output_size, line_thickness, fps)
    # frame.show()
    cv2.imshow("Image", frame)


def track(detections):
    deep_sort.predict()
    deep_sort.update(detections)
    return deep_sort.tracks


def main():
    # Use 'mixed_float16' if running on GPUs. or 'float32' on CPU
    policy = tf.keras.mixed_precision.experimental.Policy('float32')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    model = build_model()
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        process_image(frame, model, fps_print=True)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()


if __name__ == '__main__':
    main()

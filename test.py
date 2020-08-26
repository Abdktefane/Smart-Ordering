import tensorflow as tf
import numpy as np
import cv2
import hparams_config
import utils
import inference
from keras import efficientdet_keras
from trackers import sort
import datetime
import threading

D = efficientdet_keras.EfficientDetModel

model_name = 'efficientdet-d0'
image_size = '512x512'
batch_size = 1
use_xla = False
nms_score_thresh = 0.4
detection_threshold = 0.6
nms_max_output_size = 100
ckpt = "efficientdet-d0"
saved_model_dir = "savedmodeldir"
config = hparams_config.get_efficientdet_config('efficientdet-d0')
sort = sort.Sort()
coco_id_mapping = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}

def build_model() -> D:
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    config.is_training_bn = False
    config.image_size = image_size
    config.nms_configs.score_thresh = nms_score_thresh
    config.nms_configs.max_output_size = nms_max_output_size
    config.anchor_scale = [1.0, 1.0, 1.0, 1.0, 1.0]
    # Create and run the model.
    model = efficientdet_keras.EfficientDetModel(config=config)
    height, width = utils.parse_image_size(config['image_size'])
    model.build((1, height, width, 3))
    model.load_weights(ckpt)
    model.summary()
    return model


def build_model2():
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
    # TODO filter detections threshold here and max detection here or after sort
    filtered_detection = filter_detection(detections[0], detection_threshold)
    # filtered_detection = detections[0]
    if len(filtered_detection) == 0:
        return
    trackers = track(filtered_detection)
    fps = None
    if fps_print:
        elapsed_time = datetime.datetime.now() - starting_time
        fps = 1000 / (elapsed_time.total_seconds() * 1000)
        print("inference time: {}, FPS: {} ".format(elapsed_time.total_seconds() * 1000, fps))
    # threading.Thread(visualize_image(driver, frame, detections, fps)).start()
    threading.Thread(visualize_image(driver, frame, trackers, fps)).start()


def visualize_image(driver, frame, trackers, fps=None):
    # TODO modify visualize to put track id
    frame = driver.visualize(frame, trackers)
    if (fps, None):
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 30), 0, 1, (100, 100, 100), 2)
    cv2.imshow("Image", frame)


def filter_detection(detection, threshold):
    ret = []
    for d in detection:
        if d[5] > threshold:  # TODO add here filter for class
            ret.append(d)
    return np.array(ret)

def paint_overlay(image,
                  prediction,
                  label_id_mapping=None,
                  ):
    trackers_ids = prediction[:, 0]
    boxes = prediction[:, 1:5]
    classes = prediction[:, 6].astype(int)
    scores = prediction[:, 5]  # TODO remove scores from here

    label_id_mapping = label_id_mapping or coco_id_mapping
    id_mapping = parse_label_id_mapping(id_mapping)
    category_index = {k: {'id': k, 'name': id_mapping[k]} for k in id_mapping}
    img = np.array(image)

def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then this
      function assumes that the boxes to be plotted are groundtruth boxes and
      plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can be None
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    use_normalized_coordinates: whether boxes is to be interpreted as normalized
      coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw all
      boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_boxes: whether to skip the drawing of bounding boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in six.viewkeys(category_index):
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100 * scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
        if not skip_track_ids and track_ids is not None:
          if not display_str:
            display_str = 'ID {}'.format(track_ids[i])
          else:
            display_str = '{}: ID {}'.format(display_str, track_ids[i])
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        elif track_ids is not None:
          prime_multipler = _get_multiplier_for_color_randomness()
          box_to_color_map[box] = STANDARD_COLORS[(prime_multipler *
                                                   track_ids[i]) %
                                                  len(STANDARD_COLORS)]
        else:
          box_to_color_map[box] = STANDARD_COLORS[classes[i] %
                                                  len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image, box_to_instance_masks_map[box], color=color)
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image, box_to_instance_boundaries_map[box], color='red', alpha=1.0)
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=0 if skip_boxes else line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates,
          keypoint_edges=keypoint_edges,
          keypoint_edge_color=color,
          keypoint_edge_width=line_thickness // 2)

  return image


def track(detections):
    return sort.update(detections[:, 1:])


class MultiPersonTracker:
    # detector = build_model()

    # image_size = '512x512'
    # nms_score_thresh = 0.4
    # nms_max_output_size = 100
    # ckpt = 'efficientdet-d0'

    @staticmethod
    def build_model() -> D:
        config = hparams_config.get_efficientdet_config('efficientdet-d0')
        config.is_training_bn = False
        # config.image_size = MultiPersonTracker.image_size
        # config.nms_configs.score_thresh = MultiPersonTracker.nms_score_thresh
        # config.nms_configs.max_output_size = MultiPersonTracker.nms_max_output_size
        config.image_size = image_size
        config.nms_configs.score_thresh = nms_score_thresh
        config.nms_configs.max_output_size = nms_max_output_size

        config.anchor_scale = [1.0, 1.0, 1.0, 1.0, 1.0]
        # Create and run the model.
        model = efficientdet_keras.EfficientDetModel(config=config)
        height, width = utils.parse_image_size(config['image_size'])
        model.build((1, height, width, 3))
        model.load_weights(ckpt)  # TODO put checkpoint path here
        model.summary()
        return model

    @staticmethod
    def visualize_result(frame, boxes, scores, classes, valid_len):
        # Visualize results.
        length = valid_len[0]
        frame = inference.visualize_image(
            frame,
            boxes[0].numpy()[:length],
            classes[0].numpy().astype(np.int)[:length],
            scores[0].numpy()[:length],
            min_score_thresh=MultiPersonTracker.nms_score_thresh,
            max_boxes_to_draw=MultiPersonTracker.nms_max_output_size
        )
        cv2.imshow("Image", frame)

    @staticmethod
    def process_image(imgs):
        boxes, scores, classes, valid_len = MultiPersonTracker.detector(imgs, training=False, post_mode='global')
        MultiPersonTracker.visualize_result(boxes, scores, classes, valid_len)


def main():
    # Use 'mixed_float16' if running on GPUs.
    policy = tf.keras.mixed_precision.experimental.Policy('float32')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    model = build_model2()
    cap = cv2.VideoCapture(0)
    frame_id = 0
    while True:
        _, frame = cap.read()
        # MultiPersonTracker.process_image(frame)
        process_image(frame, model, fps_print=True)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()


if __name__ == '__main__':
    main()

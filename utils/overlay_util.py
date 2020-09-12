from typing import Text, Dict, Any, List, Tuple, Union
import numpy as np
import collections
import six
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

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
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGray', 'LightGrey',
    'LightPink', 'LightSalmon', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'Lime',
    'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSlateBlue',
    'MediumTurquoise', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange',
    'Orchid', 'PaleGoldenRod', 'PaleTurquoise',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple', 'RosyBrown', 'RoyalBlue', 'SaddleBrown',
    'SandyBrown', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SteelBlue',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke'
]

BACKUP_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]
FIRST_THREE_COLORS = ['red', 'yellow', 'green']


def paint_overlay(
        image,
        prediction,
        min_score_thresh=0.4,
        max_boxes_to_draw=20,
        line_thickness=6,
        skip_boxes=False,
        skip_scores=False,  # TODO make default value true
        skip_labels=False,  # TODO make default value true
        skip_track_ids=False,
        fps=None
):
    trackers_ids = prediction[:, 0]
    boxes = prediction[:, 1:5]
    classes = prediction[:, 6].astype(int)
    scores = prediction[:, 5]
    category_index = {k: {'id': k, 'name': coco_id_mapping[k]} for k in coco_id_mapping}
    img = np.array(image)
    visualize_boxes_and_labels_on_image_array(
        image=img,
        boxes=boxes,
        classes=classes,
        scores=scores,
        category_index=category_index,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw,
        line_thickness=line_thickness,
        track_ids=trackers_ids,
        skip_boxes=skip_boxes,
        skip_scores=skip_scores,
        skip_labels=skip_labels,
        skip_track_ids=skip_track_ids,
        fps=fps
    )
    return img


def _get_multiplier_for_color_randomness():
    """Returns a multiplier to get semi-random colors from successive indices.

    This function computes a prime number, p, in the range [2, 17] that:
    - is closest to len(STANDARD_COLORS) / 10
    - does not divide len(STANDARD_COLORS)

    If no prime numbers in that range satisfy the constraints, p is returned as 1.

    Once p is established, it can be used as a multiplier to select
    non-consecutive colors from STANDARD_COLORS:
    colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
    """
    num_colors = len(STANDARD_COLORS)
    prime_candidates = [5, 7, 11, 13, 17]

    # Remove all prime candidates that divide the number of colors.
    prime_candidates = [p for p in prime_candidates if num_colors % p]
    if not prime_candidates:
        return 1

    # Return the closest prime number to num_colors / 10.
    abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
    num_candidates = len(abs_distance)
    inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
    return prime_candidates[inds[0]]


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
        line_thickness=6,
        groundtruth_box_visualization_color='black',
        skip_boxes=False,
        skip_scores=False,
        skip_labels=False,
        skip_track_ids=False,
        fps=None
):
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
      fps: is the rate of frame in second, whether to draw it on screen or not.

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
                        display_str = 'ID: {}'.format(track_ids[i])
                    else:
                        display_str = '{}, ID: {}'.format(display_str, track_ids[i])
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                elif track_ids is not None:
                    prime_multipler = _get_multiplier_for_color_randomness()
                    index = int(prime_multipler * track_ids[i]) % len(STANDARD_COLORS)
                    if track_ids[i] in range(1, 4):
                        color = FIRST_THREE_COLORS[int(track_ids[i]) - 1]
                    else:
                        color = STANDARD_COLORS[index]
                    box_to_color_map[box] = color
                else:
                    box_to_color_map[box] = STANDARD_COLORS[classes[i] %
                                                            len(STANDARD_COLORS)]

    src_image = Image.fromarray(np.uint8(image)).convert('RGB')
    # Draw fps on image
    if fps is not None:
        draw_fps(src_image, fps)

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        draw_bounding_box_on_image_array(
            image=src_image,
            ymin=ymin,
            xmin=xmin,
            ymax=ymax,
            xmax=xmax,
            color=color,
            thickness=0 if skip_boxes else line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates
        )
    np.copyto(image, np.array(src_image))
    return image


def draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color='red',
        thickness=4,
        display_str_list=(),
        use_normalized_coordinates=True
):
    """Adds a bounding box to an image (numpy array).

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box (each to be shown on its
        own line).
      use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
        ymax, xmax as relative to the image.  Otherwise treat coordinates as
        absolute.
    """
    # image = Image.fromarray(np.uint8(src_image)).convert('RGB')

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line(  # draw bounding box here
            [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
            width=thickness,
            fill=color
        )
    try:
        # TODO make it cross platform
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 16, encoding="unic")
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(  # draw background box for text here
            [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],
            fill=color
        )
        draw.text(  # draw text above of background box
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font
        )
        text_bottom -= text_height - 2 * margin


def draw_fps(image, fps, fill_color='white', line_color='black', thickness=4):
    """
    draw_fps over array
    Args:
        image: the source image
        fps: is the rate of frame in second
        fill_color: the color to fill background box with
        line_color: the line color over background box
        thickness: thickness of box
    """
    fps = "FPS: {:.2f}".format(fps)
    draw = ImageDraw.Draw(image)
    fps_left = 0
    fps_bottom = 0

    try:
        # TODO make this line cross platform
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20, encoding="unic")
    except IOError:
        font = ImageFont.load_default()
    font.size = 60
    text_width, text_height = font.getsize(fps)
    total_display_str_height = (1 + 2 * 0.05) * text_height
    text_bottom = fps_bottom + total_display_str_height
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(  # draw background box for text here
        [(fps_left, text_bottom - text_height - 2 * margin), (fps_left + text_width, text_bottom)],
        fill=fill_color,
        width=thickness
    )
    draw.text(  # draw text above of background box
        (fps_left + margin, text_bottom - text_height - margin),
        fps,
        fill=line_color,
        font=font
    )
    text_bottom -= text_height - 2 * margin

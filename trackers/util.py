from numba import jit
import numpy as np
from scipy.optimize import linear_sum_assignment


@jit
def iou(boxA, boxB):
    """
    Computes Intersection Over Union between two boxes in the form [y1,x1,y2,x2]
    IOU = Area of Overlap / Area of Union
    Ps: COCO ("Common Objects in Context") Bounding box: (x-top left, y-top left, width, height)
    Ps: Pascal VOC ("Visual Object Classes") Bounding box :(x-top left, y-top left,x-bottom right, y-bottom right)
    """
    yA = np.maximum(boxA[0], boxB[0])
    xA = np.maximum(boxA[1], boxB[1])
    yB = np.minimum(boxA[2], boxB[2])
    xB = np.minimum(boxA[3], boxB[3])
    w = np.maximum(0., xB - xA)
    h = np.maximum(0., yB - yA)
    interArea = w * h
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    o = interArea / float(boxAArea + boxBArea - interArea)
    return (o)


def box_to_coco(box):
    """
    convert to COCO Bounding box format =>
    Takes a bounding box in the form [y1,x1,y2,x2,score,class] and returns z in the form
    [x,y,s,r,score,class] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio "width / height"
    """
    height = box[2] - box[0]
    width = box[3] - box[1]
    # score = box[4]
    # _class = box[5]
    x = box[1] + width / 2.
    y = box[0] + height / 2.
    scale = width * height  # scale is just area
    ratio = width / float(height)
    return np.array([x, y, scale, ratio]).reshape((4, 1))


def box_to_pascal(box, is_score: bool = False):
    """
    convert to Pascal VOC Bounding box format =>
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [y1,x1,y2,x2] where x1,y1 is the top left and x2,y2 is the bottom right
      width = square root(scale * ratio)
    """
    width = np.sqrt(box[2] * box[3])  # width = square root(scale * ratio)
    # score = box[4]
    # _class = box[5]
    height = box[2] / width  # height = scale / width
    # if is_score is False:
    return np.array(
        [box[1] - (height / 2.), box[0] - (width / 2.), box[1] + (height / 2.), box[0] + (width / 2.)]).reshape(
        (1, 4)
    )
    # else:
    # return np.array(box[1] - height / 2., [box[0] - width / 2., box[1] + height / 2.], box[0] + width / 2.,
    #                score, _class).reshape(
    #    (1, 6)
    # )


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections_indices and unmatched_trackers
    """
    if len(trackers) == 0:  # when first call return index of detections in unmatched_detections and empty for else
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    # create cost matrix (complexity is n*n for iou)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # matched_indices = linear_assignment(-iou_matrix)
    # linear_sum_assignment is the Hungarian algorithm
    # ps: result will be a 2x2 array first column represent index of detections and second column index of tracker
    #    and each line is represent one object
    matched_indices = np.array(list(zip(*linear_sum_assignment(-iou_matrix))))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:  # give me all detections matched indices
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:  # give me all trackers matched indices
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

from numba import jit
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Text, Tuple, Union
from trackers.kalman_filter import chi2inv95
#from sklearn.utils.linear_assignment_ import linear_assignment  # TODO check if that solve with scipy.optimize

INFINITY_COST = 1e+5


def box_to_coco(box):
    """
    convert to COCO Bounding box format => Takes a bounding box in [y_min,x_min,y_max,x_max] format and returns[x,y,s,r]
    format, where x,y is the centre of the box and s is the scale/area and r is the aspect ratio "width / height"
    """
    height = box[2] - box[0]
    width = box[3] - box[1]
    x = box[1] + width / 2.
    y = box[0] + height / 2.
    scale = width * height  # scale is just area
    ratio = width / float(height)
    # return np.array([x, y, scale, ratio]).reshape((4, 1))
    # return np.c_[x, y, scale, ratio]
    return [x, y, scale, ratio]


def box_to_pascal(box):
    """
    convert to Pascal VOC Bounding box format => Takes a bounding box in the centre form [x,y,s,r]
    and convert it to [y_min,x_min,y_max,x_max] format, where width = square root(scale * ratio)
    """
    width = np.sqrt(box[2] * box[3])  # width = square root(scale * ratio)
    height = box[2] / width  # height = scale / width
    return np.c_[box[1] - (height / 2.), box[0] - (width / 2.), box[1] + (height / 2.), box[0] + (width / 2.)]


def parse_image_size(image_size: Union[Text, int, Tuple[int, int]]):
    """Parse the image size and return (height, width).

    Args:
      image_size: A integer, a tuple (H, W), or a string with HxW format.

    Returns:
      A tuple of integer (height, width).
    """
    if isinstance(image_size, int):
        # image_size is integer, with the same width and height.
        return image_size, image_size

    if isinstance(image_size, str):
        # image_size is a string with format WxH
        width, height = image_size.lower().split('x')
        return int(height), int(width)

    if isinstance(image_size, tuple):
        return image_size

    raise ValueError('image_size must be an int, WxH string, or (height, width)'
                     'tuple. Was %r' % image_size)


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
        A bounding box in format (y_min,x_min,y_max,x_max).
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox = bbox[0]
    bbox_tl = bbox[:2]
    bbox_br = bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, 2:]

    tl = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],  # max between y in top left
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]  # max between x in top left
    ]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],  # max between y in bottom right
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]  # max between x in bottom right
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_candidates = [(c[2] - c[0]) * (c[3] - c[1]) for c in candidates]
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(
        tracks,
        detections,
        track_indices=None,
        detection_indices=None
):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INFINITY_COST
            continue

        bbox = tracks[track_idx].to_base()
        candidates = np.asarray([detections[i].to_base() for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix


def gate_cost_matrix(
        kf,
        cost_matrix,
        tracks,
        detections,
        track_indices,
        detection_indices,
        gated_cost=INFINITY_COST,
        only_position=False
):
    """
    Invalidate infeasible entries in cost matrix(apply gate equation on it) based on the state
    distributions obtained by Kalman filtering.
    if bigger than gate value replace it with INFINITY_COST

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        mah_distance = kf.mah_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, mah_distance > gating_threshold] = gated_cost
    return cost_matrix


def min_cost_matching(
        distance_metric,
        max_distance,
        tracks,
        detections,
        track_indices=None,
        detection_indices=None
):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.
    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5  # TODO search what happen here.
    # indices = linear_assignment(cost_matrix) # Old method
    # indices = linear_sum_assignment(cost_matrix)
    indices = np.array(list(zip(*linear_sum_assignment(cost_matrix))))

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric,
        max_distance,
        cascade_depth,
        tracks,
        detections,
        track_indices=None,
        detection_indices=None
):
    """
    Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric=distance_metric,
            max_distance=max_distance,
            tracks=tracks,
            detections=detections,
            track_indices=track_indices_l,
            detection_indices=unmatched_detections
        )
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


'''
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
            iou_matrix[d, t] = iou(det.as_np(), trk.as_np())

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

'''

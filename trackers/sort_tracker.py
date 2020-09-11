import numpy as np
from trackers.tracker import Tracker
from trackers.kalman_filter import KalmanFilter
from utils.hyper_params import default_params
from utils import util


class Sort(Tracker):
    def __init__(
            self,
            min_score_thresh=default_params['min_score_thresh'],
            max_age=default_params['max_age'],
            nn_init=default_params['nn_init'],
            filter=KalmanFilter()
    ):
        super(Sort, self).__init__(
            min_score_thresh=min_score_thresh,
            max_age=max_age,
            nn_init=nn_init,
            filter=filter
        )

    def _match(self, detections):
        # return util.associate_detections_to_trackers(detections, self.tracks, self.min_score_thresh)
        return util.min_cost_matching(
            distance_metric=util.iou_cost,
            max_distance=self.min_score_thresh,
            tracks=self.tracks,
            detections=detections
        )

    # detections: a list of vector, with each vector has the format of [ymin, xmin, ymax, xmax, score, class]
    def _update(self, detections):
        """
        Params:
        detections - a numpy array of Detections .
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        super(Sort, self)._update(detections)
        # return self.tracks
        return np.array([trk.to_visualize() for trk in self.tracks]).reshape(-1, 7)

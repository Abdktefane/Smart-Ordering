import numpy as np
from trackers.tracker import Tracker
from trackers.kalman_filter import KalmanFilter
from utils.hyper_params import default_params
from utils import util
from utils import nn_matching
from utils import features_util


# TODO add feature extractor here and put it in (_modify_detections) (DONE)
class DeepSort(Tracker):
    def __init__(
            self,
            max_cosine_distance=default_params['max_cosine_distance'],
            nn_budget=default_params['nn_budget'],
            min_score_thresh=default_params['min_score_thresh'],
            max_age=default_params['max_age'],
            nn_init=default_params['nn_init'],
            filter=KalmanFilter()
    ):
        """"
        @param max_cosine_distance: float
            The cosine distance threshold. Samples with larger distance are considered an
            invalid match.
        @param nn_budget: Optional[int]
            gallery size, If not None, fix samples per class to at most this number. Removes
            the oldest samples when the budget is reached.
        @param min_score_thresh: float
            iou threshold
        @param max_age: int
            The maximum number of consecutive misses before the track state is
            set to `Deleted`.
        @param nn_init: int
            Number of consecutive detections before the track is confirmed. The
            track state is set to `Deleted` if a miss occurs within the first
            `n_init` frames.
        @param filter: trackers.kalman_filter.KalmanFilter
            mathematical idea to predict state of observed object.

        """
        super(DeepSort, self).__init__(
            min_score_thresh=min_score_thresh,
            max_age=max_age,
            nn_init=nn_init,
            filter=filter
        )
        # create the feature extractor model
        self.feature_extractor = features_util.create_box_encoder()
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            """"
            calculate cost matrix for specific track_indices and detection_indices and apply gate equation on it

            Parameters
            ----------
            tracks : List[track.Track]
                A list of predicted tracks at the current time step.
            dets : List[detection.Detection]
                A list of detections at the current time step.
            track_indices : List[int]
                List of track indices that maps rows in `cost_matrix` to tracks in
                `tracks` (see description above).
            detection_indices : List[int]
                List of detection indices that maps columns in `cost_matrix` to
                detections in `detections` (see description above).
            Returns
            -------
            ndarray
                Returns the gated cost matrix.
            """
            # get features from all detections
            features = np.array([dets[i].feature for i in detection_indices])
            # get track id  from all trackers
            targets = np.array([tracks[i].track_id for i in track_indices])
            # the cost"cosine distance or cosine similarity" between each track_id and each feature
            # without gated with max_cosine_distance
            cost_matrix = self.metric.distance(features, targets)
            # measure mah distance and gated it and modify cost matrix
            cost_matrix = util.gate_cost_matrix(
                kf=self.filter,
                cost_matrix=cost_matrix,
                tracks=tracks,
                detections=dets,
                track_indices=track_indices,
                detection_indices=detection_indices
            )
            # the final cost matrix with cosine distance and  mah_distance gated equation
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # apply gated metrics to confirmed tracks in matching cascade algorithm.
        matches_a, unmatched_tracks_a, unmatched_detections = util.matching_cascade(
            distance_metric=gated_metric,
            max_distance=self.metric.matching_threshold,
            cascade_depth=self.max_age,
            tracks=self.tracks,
            detections=detections,
            track_indices=confirmed_tracks
        )

        # Associate remaining tracks together with unconfirmed_tracks_a (from output of matching_cascade) using IOU.
        # This helps to to account for sudden appearance changes, e.g.,
        # due to partial occlusion with static scene geometry, and to
        # increase robustness against erroneous initialization.
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if
                                                     self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = util.min_cost_matching(
            distance_metric=util.iou_cost,
            max_distance=self.min_score_thresh,
            tracks=self.tracks,
            detections=detections,
            track_indices=iou_track_candidates,
            detection_indices=unmatched_detections
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    # detections: a list of vector, with each vector has the format of [ymin, xmin, ymax, xmax, score, class]
    def _update(self, detections):
        """
        Params:
        detections - a numpy array of Detections .
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        super(DeepSort, self)._update(detections)
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)
        return np.array([trk.to_visualize() for trk in self.tracks]).reshape(-1, 7)

    def _extract_features(self, detections, frame):
        """
        feed frame to feature_extractor model for extract feature for each detections
        and add to it .

        @param detections: List[detectors.detection.Detection]
            A list of detections at the current time step.
        @param frame: ndarray
            the input image
        @return: List[detectors.detection.Detection]
        """
        features = self.feature_extractor(frame, [d.to_tlbr() for d in detections])
        # detections = [det.set_feature(feature) and det for det, feature in zip(detections, features)]
        featured_detection = []
        for det, feature in zip(detections, features):
            det.set_feature(feature)
            featured_detection.append(det)
        return featured_detection

    def _modify_detections(self, detections, frame=None):
        return self._extract_features(detections, frame)

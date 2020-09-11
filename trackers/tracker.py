from abc import ABC, abstractmethod
from trackers.track import Track


class Tracker(ABC):
    """
    This class represents a blueprint for other trackers models and must adopt it.

    Parameters
    ----------

    filter: KalmanFilter
        mathematical idea to predict state of observed object.

    Attributes
    ----------

    """

    def __init__(self, min_score_thresh, max_age, nn_init, filter):
        self.min_score_thresh = min_score_thresh
        self.max_age = max_age
        self.nn_init = nn_init
        self.tracks = []
        self.frame_count = 0
        self.filter = filter
        self._next_id = 1

    def __call__(self, detections, frame=None):
        detections = self._modify_detections(detections, frame)
        return self._update(detections)

    def _predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.filter)

    def _update(self, detections):
        """
        Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        self.frame_count += 1
        self._predict()
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        # return matches, unmatched_tracks, unmatched_detections
        # Update track set.
        print("matches = {} \n unmatched_tracks = {} \n unmatched_detections = {}------------\n\n".format(len(matches),
                                                                                          len(unmatched_tracks),
                                                                                          len(unmatched_detections)))
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.filter, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        deleted_tracks = [t for t in self.tracks if t.is_deleted()]  # TODO better way for complexity
        self.tracks = [t for t in self.tracks if not t.is_deleted()]  # TODO modify this to manage track id here (DONE)
        self._resort_ids(self.tracks, deleted_tracks)
        return deleted_tracks

    def _match(self, detections):
        pass

    def _initiate_track(self, detection):
        mean, covariance = self.filter.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean=mean,
            covariance=covariance,
            track_id=self._next_id,
            n_init=self.nn_init,
            max_age=self.max_age,
            confidence=detection.confidence,
            class_=detection.class_,
            feature=detection.feature
        ))
        self._next_id += 1

    def _modify_detections(self, detections, frame=None):
        return detections

    def _resort_ids(self, active_tracks, deleted_tracks):
        for active in active_tracks:
            for del_tracks in deleted_tracks:
                if active.track_id >= del_tracks.track_id:
                    active.track_id -= 1
        self._next_id -= len(deleted_tracks)

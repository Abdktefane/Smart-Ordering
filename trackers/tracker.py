from abc import ABC, abstractmethod
from trackers.track import Track


class Tracker(ABC):
    """
    This class represents a blueprint for other trackers models and must adopt it.

    Parameters
    ----------

    filter: trackers.kalman_filter.KalmanFilter
        mathematical idea to predict state of observed object.
    nn_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.

    Attributes
    ----------
    tracks : list[trackers.track.Track]
        list of tracks that trackers contain
    _next_id : int
        unique id for incoming track to associate with it
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
        detections : List[detectors.detection.Detection]
            A list of detections at the current time step.

        """
        self.frame_count += 1
        self._predict()
        # return matches, unmatched_tracks, unmatched_detections
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.filter, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        deleted_tracks = [t for t in self.tracks if t.is_deleted()]  # TODO better way for complexity (DONE)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]  # TODO modify this to manage track id here (DONE)
        self._resort_ids(self.tracks, deleted_tracks)

    def _match(self, detections):
        """
        abstract function for solve association problem between tracks and detections .
        ps: each child must override with proper algorithm and return three lists with respect to sorting .

        @param detections: List[detectors.detection.Detection]
            A list of detections at the current time step.
        @return: (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
        """
        return [0][0][0]

    def _initiate_track(self, detection):
        """
        initialize new track and add it to tracks list

        @param detection: detectors.detection.Detection
            detections object at the current time step.
        """
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

    def _resort_ids(self, active_tracks, deleted_tracks):
        """
        call when any track leave the scene(dead state).
        resort tracks id's  and modify next_id for next incoming tracks
        @param active_tracks: list[trackers.track.Track]
            tracks that steel in scene and need to modify their id's
        @param deleted_tracks: list[trackers.track.Track]
            tracks that deleted from scene
        """
        for active in active_tracks:
            for del_tracks in deleted_tracks:
                if active.track_id >= del_tracks.track_id:
                    active.track_id -= 1
        self._next_id -= len(deleted_tracks)

    def _modify_detections(self, detections, frame=None):
        """
        any child need to modify detections before using it should override this method .
        ps: if no need modify detections don't override

        @param detections: List[detectors.detection.Detection]
            A list of detections at the current time step.
        @param frame: ndarray
            the input image
        @return: List[detectors.detection.Detection]
        """
        return detections

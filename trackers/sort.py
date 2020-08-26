from __future__ import print_function
import numpy as np
from trackers import util
from trackers import kalman


# from trackers.util import associate_detections_to_trackers
# from trackers.kalman import KalmanBoxTracker


class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    # detections: a list of vector, with each vector has the format of [ymin, xmin, ymax, xmax, score, class]
    def update(self, detections):
        """
        Params:
        detections - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trackers = np.zeros((len(self.trackers), 6))  # [ymin, xmin, ymax, xmax, score, class]
        to_delete = []
        ret = []

        for t, trk in enumerate(trackers):
            pos = self.trackers[t].predict()[0]  # predict new value
            # TODO add that ==> trk[:] = [pos[0], pos[1], pos[2], pos[3], 0 , Person class index]
            # trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):  # if position is not a number add index to delete later
                to_delete.append(t)

        # TODO: for learning
        trackers = np.ma.compress_rows(np.ma.masked_invalid(trackers))

        for t in reversed(to_delete):  # clean up trackers
            self.trackers.pop(t)  # TODO: here we should manage the id process

        matched, unmatched_detections, unmatched_trackers = util.associate_detections_to_trackers(detections, trackers)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trackers:  # search for detection line in based on track index
                detections_index = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(detections[detections_index, :][0])  # TODO check why this [0] exist here

        # create and initialise new trackers for unmatched detections
        for i in unmatched_detections:
            trk = kalman.KalmanBoxTracker(detections[i, :])  # TODO modify KalmanBoxTracker index
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
                ret.append(np.concatenate(([trk.id + 1], d), axis=None).reshape(1,
                                                                                -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret).reshape(-1, 7)  # return in flat format
            # return np.array(ret)
        return np.empty((0, 7))  # [track_id, ymin, xmin, ymax, xmax, score, class]

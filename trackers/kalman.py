from filterpy.kalman import KalmanFilter
import numpy as np
# from trackers.util import box_to_coco, box_to_pascal
from trackers import util

"""
This class represents the internal state of individual tracked objects observed as bbox.
"""


# TODO check if that line used KalmanBoxTracker().kf.x to get state and replace with KalmanBoxTracker().get_state()
class KalmanBoxTracker(object):
    count = 0

    @staticmethod
    def minus_id():
        KalmanBoxTracker.count -= 1

    """
    Initialises a tracker using initial bounding box.
    """

    def __init__(self, bbox):

        """
        define constant velocity model x = [u, v, s, r, u˙, v˙, s˙]
        where u and v represent the horizontal and vertical pixel location of the centre of the target,
        while the scale s and r represent the scale (area) and the aspect ratio of the target’s bounding box.
        ps: dim_x =  size of the state vector
        ps: dim_z =  size of the measurement vector
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # define state transition matrix 7x7 "A"
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1]
            ]
        )

        """
        define Measurement function 4x7 "H in equation #5 in update step" 
        ps: 4*7 matrix multiply 7*1 => 4x1 matrix
        """
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ]
        )

        """
        define the measurement noise ("default value is eye(dim_z)") =>
         [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]]

        after [2:, 2:] *= 10 => 
        [[ 1.  0.  0.  0.]
        [ 0.  1.  0.  0.]
        [ 0.  0. 10.  0.]
        [ 0.  0.  0. 10.]]                                                               
        """
        self.kf.R[2:, 2:] *= 10.

        """
        define the covariance matrix  ("default value is eye(dim_x)") =>
         [[1. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 0. 1.]]

        after [4:, 4:] *= 1000 => 
        [[   1.    0.    0.    0.    0.    0.    0.]
         [   0.    1.    0.    0.    0.    0.    0.]
         [   0.    0.    1.    0.    0.    0.    0.]
         [   0.    0.    0.    1.    0.    0.    0.]
         [   0.    0.    0.    0. 1000.    0.    0.]
         [   0.    0.    0.    0.    0. 1000.    0.]
         [   0.    0.    0.    0.    0.    0. 1000.]]

         after P *= 10 =>
         [[   10.     0.     0.     0.     0.     0.     0.]
         [    0.    10.     0.     0.     0.     0.     0.]
         [    0.     0.    10.     0.     0.     0.     0.]
         [    0.     0.     0.    10.     0.     0.     0.]
         [    0.     0.     0.     0. 10000.     0.     0.]
         [    0.     0.     0.     0.     0. 10000.     0.]
         [    0.     0.     0.     0.     0.     0. 10000.]]
         
        "Since the velocity is unobserved
        at this point the covariance of the velocity component is initialised with large values"
        """
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.

        # define the process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # define initial state
        self.kf.x[:4] = util.box_to_coco(bbox)  # TODO save here the score and class score = bbox[4], _class =  bbox[5]

        #
        self.time_since_update = 0

        # assign new id
        self.id = KalmanBoxTracker.count

        # increment id pool
        KalmanBoxTracker.count += 1

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = bbox[4]
        self._class = bbox[5]

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = bbox[4]
        self._class = bbox[5]
        self.kf.update(util.box_to_coco(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        x = [u, v, s, r, u˙, v˙, s˙]
           [ 0, 1, 2, 3, 4 , 5 , 6 ]
        """
        print("start of KalmanBoxTracker.predict")
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(util.box_to_pascal(self.kf.x))  # TODO add here the score and the class
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return np.concatenate((util.box_to_pascal(self.kf.x).reshape(1, -1), [self.score], [self._class]), axis=None)

    def decrease_id(self):
        self.id -= 1

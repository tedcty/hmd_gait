from enum import Enum
import copy
import numpy as np
import pandas as pd

from ptb.core import Yatsdo
from ptb.ml.ml_util import MLOperations, MLKeys


class MetaMarkerSet(Enum):
    torso = ["L_Acromion", "R_Acromion", "Sternum"]
    pelvis = ["R_ASIS", "L_ASIS", "R_PSIS", "L_PSIS"]
    left_upper_arm = ["L_Lat_HumEpicondyle", "L_Med_HumEpicondyle"]
    right_upper_arm = ["R_Lat_HumEpicondyle", "R_Med_HumEpicondyle"]

    left_upper_leg = ["L_MedKnee", "L_LatKnee"]
    right_upper_leg = ["R_MedKnee", "R_LatKnee"]

    left_lower_leg = ["L_FibHead", "L_MidShank", "L_MedAnkle", "L_LatAnkle"]
    right_lower_leg = ["R_FibHead", "R_MidShank", "R_MedAnkle", "R_LatAnkle"]

    @staticmethod
    def get(m):
        return copy.deepcopy(m.value)

class EventTypes(Enum):
    standing = 0 # stationary for at least 5 seconds
    walking = 1 # straight line walking with at least 3 strides
    turning = 2 # turning for more than 20 degrees
    stairs_ascent = 3 # step up at least one step
    stairs_descent = 4 # step down at least one step
    step_over = 5


class GaitMLClassifier:
    def __init__(self):
        pass

    def train(self, x, y):
        pass

class FeatureSet:
    def __init__(self):
        self.features = {}
        self.cache = {}

    @staticmethod
    def moving_windows(data:Yatsdo, size=60):
        """
        This static method creates a moving window of data for feature extraction
        :param data:  data as a Yatsdo object
        :param size: the number of frames of data
        :return: a list of windows
        """
        # 1 sec window due to imu sampling rate @ 60Hz
        num_winds = [[s, s + size] for s in range(data.shape[0] - size)]
        windows = []
        for s in num_winds:
            windows.append(data.get_samples(data.data[s[0]:s[1], 0]))
        return windows

    @staticmethod
    def windows_to_dataframe(ref:Yatsdo, windows):
        """
        This static method converts the list of the windows to a single data frame for feature extraction
        :param ref: the reference Yatsdo object that was windowed
        :param windows: the list of windows to combine
        :return: a data frame of windows with the included window ids
        """
        cols = ref.column_labels
        cols.insert(0, 'id')
        window_id = 1
        updated_windows = []
        for w in windows:
            n = np.zeros([w.shape[0], w.shape[1]+1])
            n[:, 0] = window_id
            window_id += 1
            updated_windows.append(n)
        nstack = np.vstack(updated_windows)
        return pd.DataFrame(data=nstack, columns=cols)

    def extract(self, data_name: str, windows:pd.DataFrame):
        """
        This method extracts one set of windows at a time with the associate name
        :param data_name: a given name for the windows
        :param windows: a dataframe of windows as outlined by tsfresh
        :return: None, this method save the features in a dictionary
        """
        efx, param = MLOperations.extract_features_from_x(windows, fc_parameters=MLKeys.MFCParameters)
        self.features[data_name] = [efx, param]




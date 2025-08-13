import pandas as pd
import numpy as np

from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from ptb.util.data import StorageIO
from ptb.util.math.filters import Butterworth


class Sync:
    def __init__(self, ref):
        self.ik = ref

    def by_peaks(self, root:str=None, visualise=False):
        """

        :param root: the folder where the imu data is.
        :param visualise: plot the time-series
        :return: the offset
        """
        rhand = pd.read_csv("{0}RightHand_imu_ori.csv".format(root))
        lhand = pd.read_csv("{0}LeftHand_imu_ori.csv".format(root))
        rfore = pd.read_csv("{0}RightHand_imu_ori.csv".format(root))
        lfore = pd.read_csv("{0}LeftHand_imu_ori.csv".format(root))

        def pre_process_imu_peaks(hand):
            hand_abs = np.abs(hand)  # force quaternion to the same quarter for peak detection
            # only no used as the actual angle
            hand_abs.iloc[:, 0] = np.array([i * 1 / 60 for i in range(0, hand_abs.shape[0])])
            hand_norm = []
            for i in range(0, hand_abs.shape[0]):
                r = Rotation.from_quat(hand_abs.iloc[i, 1:], scalar_first=True)
                e = r.as_euler('xyz', True)
                hand_norm.append(e[0])

            hand_norm = np.array(hand_norm)
            hand_norm = np.abs(hand_norm - np.max(hand_norm))
            hand_abs_norm = np.zeros([hand_abs.shape[0], 2])
            hand_abs_norm[:, 0] = hand_abs.iloc[:, 0]
            hand_abs_norm[:, 1] = hand_norm
            return hand_abs_norm

        rhand_abs_norm = pre_process_imu_peaks(rhand)
        lhand_abs_norm = pre_process_imu_peaks(lhand)
        sto = StorageIO.load(self.ik)
        hz = 100
        sync = sto.data[['time', 'arm_add_l', 'arm_add_r']]
        ar1, _ = find_peaks(rhand_abs_norm[:, 1], height=np.max(np.abs(rhand_abs_norm[:, 1])) * 0.60)
        ar2, _ = find_peaks(lhand_abs_norm[:, 1], height=np.max(np.abs(lhand_abs_norm[:, 1])) * 0.60)
        imu_average = np.array([rhand_abs_norm[int((ar1[i] + ar2[i]) / 2.0), 0] for i in range(0, 2)])
        for i in range(1, 3):
            sync.iloc[:, i] = Butterworth.butter_low_filter(data=sync.iloc[:, i], fs=hz, cut=3)
            sync.iloc[:, i] = sync.iloc[:, i] / np.max(sync.iloc[:, i])
        sync_abs = np.abs(sync)
        sync_abs_np = sync_abs.to_numpy()
        ar3, _ = find_peaks(sync_abs_np[:, 1], height=np.max(np.abs(sync_abs_np[:, 1])) * 0.60)
        ar4, _ = find_peaks(sync_abs_np[:, 2], height=np.max(np.abs(sync_abs_np[:, 2])) * 0.60)
        mocap_average = np.array([sync_abs_np[int((ar3[i] + ar4[i]) / 2.0), 0] for i in range(0, 2)])
        offset = np.mean((imu_average - mocap_average))
        frames_offset = offset / (1 / 60)
        rhand_abs_norm[:, 0] = rhand_abs_norm[:, 0] - offset
        lhand_abs_norm[:, 0] = lhand_abs_norm[:, 0] - offset
        if visualise:
            plt.figure()
            plt.plot(sync_abs.iloc[:, 0], sync_abs.iloc[:, 1])
            plt.plot(sync_abs_np[ar3, 0], sync_abs_np[ar3, 1], 'o')
            plt.plot(sync_abs.iloc[:, 0], sync_abs.iloc[:, 2])
            plt.plot(sync_abs_np[ar4, 0], sync_abs_np[ar4, 2], 'o')
            plt.plot(rhand_abs_norm[:, 0], rhand_abs_norm[:, 1])
            plt.plot(rhand_abs_norm[ar1, 0], rhand_abs_norm[ar1, 1], 'x')
            plt.plot(lhand_abs_norm[:, 0], lhand_abs_norm[:, 1])
            plt.plot(lhand_abs_norm[ar2, 0], lhand_abs_norm[ar2, 1], 'x')
            plt.show()
        print("Offset: {0}s".format(offset))
        return offset
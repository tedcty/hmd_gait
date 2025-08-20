import numpy as np
import pandas as pd
from dask.array import median
from ptb.core import Yatsdo
from util import model, io
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def preprocess():
    rootx = "M:/Mocap/Movella_Re/P025/Straight Normal 01/"
    ik = None
    s = io.Sync(ik)
    s.by_peaks(rootx)
    pass

def model_creation(imu_n, imu_df, evnt, wnsize):
    data = Yatsdo(imu_df)
    x = data.x - data.x[0]
    data.data[:, 0] = x
    data.update()
    d = int(np.floor((x[-1] - x[0]) / (1 / 100)))
    t = [i * (1 / 100) for i in range(0, d)]
    data_resample = Yatsdo(data.get_samples(t, as_pandas=True))
    window_list = model.FeatureSet.moving_windows(data_resample, size=wnsize)
    window_pos = []
    window_neg = []
    def check_is_pos(k):
        for e in evnt:
            if e.startswith('p'):
                if evnt[e][0] < k < evnt[e][1]:
                    return True
        return False
    for w in window_list:
        if check_is_pos(w[0, 0]) and check_is_pos(w[-1, 0]):
            window_pos.append(w)
        else:
            window_neg.append(w)
    window_df_pos = model.FeatureSet.windows_to_dataframe(data_resample, window_pos)
    window_df_neg = model.FeatureSet.windows_to_dataframe(data_resample, window_neg)
    f = model.FeatureSet()
    f.extract(imu_n + 'pos', window_df_pos)
    f.extract(imu_n + 'neg', window_df_neg)
    pos = f.features[imu_n + 'pos'][0].to_numpy()
    pos_ones = np.ones([1, pos.shape[0]]).T
    neg = f.features[imu_n + 'neg'][0].to_numpy()
    neg_ones = np.zeros([1, neg.shape[0]]).T
    to_X = np.vstack([pos, neg])
    to_y = np.vstack([pos_ones, neg_ones])
    # f.feature_select(evnt)
    clf = model.GaitMLClassifier()
    clf.train(pd.DataFrame(data=to_X, columns=f.features[imu_n + 'pos'][0].columns), to_y)
    p = clf.clf.feature_importances_
    s = pd.Series(data=p, index=f.features[imu_n + 'pos'][0].columns)
    s1 = s.sort_values(ascending=False)
    # return clf
    pass

def test_event(x0):
    ret = {'p0': [x0[283], x0[639], 1],
           'p1': [x0[787], x0[1057], 1],
           'n0': [x0[0], x0[283], 0],
           'n1': [x0[639], x0[787], 0],
           'n2': [x0[1057], x0[1293], 0]}
    return ret

def test_check(x, t):
    for y in x:
        if y.startswith('n'):
            z = x[y][:2]
            if (t > z[0]) and (t < z[1]):
                return True
    return False


if __name__ == '__main__':
    # preprocess()
    root = "M:/Mocap/Movella_Re/P025/Straight Normal 01/"
    #event = pd.read_csv("{0}trial_event.csv".format(root))
    imujoint = pd.read_csv("{0}joint_angles_2.csv".format(root))
    test_label = ['jRightKnee_Z', 'jLeftKnee_Z']
    x = imujoint['time'].to_numpy()
    x = x - x[0]
    event = test_event(x)
    hz = 1.0/float(np.mean(x[1:] - x[:-1]))
    imujoint['time'] = x

    a = imujoint[['time', test_label[0], test_label[1]]]
    a_yd = Yatsdo(a)
    d = int(np.floor((x[-1]-x[0])/(1/100)))
    t = [i*(1/100) for i in range(0, d)]
    ma_yd = a_yd.to_panda_data_frame()
    na_yd = a_yd.get_samples(t, as_pandas=True)
    # plt.figure()
    # plt.plot(ma_yd['time'], ma_yd[test_label[0]])
    # plt.plot(ma_yd['time'], ma_yd[test_label[1]])

    ar1a, _ = find_peaks(ma_yd[test_label[0]], height=np.max(np.abs(ma_yd[test_label[0]])) * 0.60)
    ar2a, _ = find_peaks(ma_yd[test_label[1]], height=np.max(np.abs(ma_yd[test_label[1]])) * 0.60)
    # plt.plot(ma_yd['time'][ar1a], ma_yd[test_label[0]][ar1a], 'o')
    # plt.plot(ma_yd['time'][ar2a], ma_yd[test_label[1]][ar2a], 'o')
    #
    # plt.plot(na_yd['time'], na_yd[test_label[0]])
    # plt.plot(na_yd['time'], na_yd[test_label[1]])

    ar1, _ = find_peaks(na_yd[test_label[0]], height=np.max(np.abs(na_yd[test_label[0]])) * 0.60)
    ar2, _ = find_peaks(na_yd[test_label[1]], height=np.max(np.abs(na_yd[test_label[1]])) * 0.60)
    # plt.plot(na_yd['time'][ar1], na_yd[test_label[0]][ar1], 'x')
    # plt.plot(na_yd['time'][ar2], na_yd[test_label[1]][ar2], 'x')
    # plt.show()
    arT = []
    start = na_yd['time'][ar1].to_numpy()
    follow = na_yd['time'][ar2].to_numpy()
    if start[0] > follow[0]:
        start = na_yd['time'][ar2].to_numpy()
        follow = na_yd['time'][ar1].to_numpy()
    for i in range(0, max([len(ar1), len(ar2)])):
        if i < len(start):
            arT.append(start[i])
        if i < len(follow):
            arT.append(follow[i])

    arT2 = []
    start = ma_yd['time'][ar1a].to_numpy()
    follow = ma_yd['time'][ar2a].to_numpy()
    if start[0] > follow[0]:
        start = ma_yd['time'][ar2a].to_numpy()
        follow = ma_yd['time'][ar1a].to_numpy()
    for i in range(0, max([len(ar1a), len(ar2a)])):
        if i < len(start):
            arT2.append(start[i])
        if i < len(follow):
            arT2.append(follow[i])
    steps = 5
    dt_list = []
    dt_list2 = []
    for i in range(0, len(arT)-(steps-1)):
        dt = arT[i + (steps-1)] - arT[i]
        dt2 = arT2[i + (steps - 1)] - arT2[i]
        dt_list.append(dt)
        dt_list2.append(dt2)
        print('[{0}, {1}]: {2}'.format(arT[i], arT[i + (steps-1)], dt))
    median = np.median(dt_list)
    upper_quartile = np.quantile(dt_list, 0.90)
    lower_quartile = np.quantile(dt_list, 0.10)
    dt_list_outlier_rm = []
    for i in dt_list:
        if lower_quartile <= i <= upper_quartile:
            dt_list_outlier_rm.append(i)

    window = float(np.mean(dt_list_outlier_rm)) + float(np.std(dt_list_outlier_rm))

    median2 = np.median(dt_list2)
    upper_quartile2 = np.quantile(dt_list2, 0.90)
    lower_quartile2 = np.quantile(dt_list2, 0.10)
    dt_list_outlier_rm2 = []
    for i in dt_list2:
        if lower_quartile2 <= i <= upper_quartile2:
            dt_list_outlier_rm2.append(i)

    window2 = float(np.mean(dt_list_outlier_rm2)) + float(np.std(dt_list_outlier_rm2))
    imu_name = 'RightHand'
    #imu_1 = pd.read_csv("{0}{1}_imu_ori.csv".format(root, imu_name))
    imu_1 = pd.read_csv("{0}{1}_imu_vec3_2_raw.csv".format(root, imu_name))
    winsize = int(np.round(window2*100, 0))
    lower = model_creation(imu_name, imu_1, event, winsize)
    pass
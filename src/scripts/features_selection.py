import copy

import numpy as np
import pandas as pd
import gc
import os
import shutil
from enum import Enum

from numpy.polynomial import Polynomial
from itertools import permutations
from scipy import interpolate

import ptb
from ptb.core import Yatsdo
from ptb.util.io.helper import StorageIO, StorageType, TRC, JSONSUtl
from ptb.util.gait.analysis import Analysis
from ptb.util.math.filters import Butterworth
from ptb.util.math.transformation import PCAModel, Cloud
from ptb.ml.ml_util import MLOperations
from ptb.util.osim.osim_model import IK
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, resample_poly
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class Poop:
    def __init__(self, data, x=None):
        self.hz = 60
        dataraw = np.zeros([data.shape[0], data.shape[1] + 1])
        dataraw[:, 0] = [i/self.hz for i in range(dataraw.shape[0])]
        if isinstance(data, pd.DataFrame):
            dataraw[:, 1:] = data.to_numpy()
            col = ['time']
            for c in data.columns:
                col.append(c)
            datapd = pd.DataFrame(data=dataraw, columns=col)
        else:
            dataraw[:, 1:] = data
            datapd = pd.DataFrame(data=dataraw)

        self.imu = ptb.core.Yatsdo(datapd)
        self.windows = None


    def ik_runner(self):

        pass

    def pipe(self):
        pass

    def window(self, size=60):
        # 1 sec
        num_winds = [[s, s+size] for s in range(self.imu.shape[0] - size)]
        windows = []
        for s in num_winds:
            print(s)
            windows.append(self.imu.get_samples(self.imu.data[s[0]:s[1], 0]))
        self.windows = windows
        return windows

    @staticmethod
    def loader(file):
        ret = None
        if file.endswith(".pkl"):
            dat = np.load(file, allow_pickle=True)
            ret = Poop(dat)
        if file.endswith(".csv"):
            ret = Poop(pd.read_csv(file))
        return ret

    @staticmethod
    def sync(root, ik, visualise=False):

        rhand = pd.read_csv("{0}RightHand_imu_ori.csv".format(root))
        lhand = pd.read_csv("{0}LeftHand_imu_ori.csv".format(root))

        def pre_process_imu_peaks(hand):
            hand_abs = np.abs(hand)     # force quaternion to the same quarter for peak detection
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
        sto = StorageIO.load(ik)
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

    @staticmethod
    def normal_model_v1():
        wk_dir = "M:/Mocap/P005/Session 1/"
        trial = "Straight normal 2.c3d"
        template = "M:/template/Straight normal 1.xml"
        model = "M:/Models/P005.osim"
        gc.collect()
        print(ptb.core.info.all)
        root = "M:/Mocap/Movella_Re/P005/Straight normal 2/"

        def is_lower(k):
            for p in ['foot', 'lowerleg', 'upperleg', 'pelvis']:
                if p in k.lower():
                    return True
            return False

        ik = r"E:\test\test.mot"
        osimmodel = r"M:\Models\P005.osim"
        input_dir = "E:/test/"
        offset = Poop.sync(root, ik=ik, visualise=True)
        pooper = [f for f in os.listdir(root) if is_lower(f) and f.endswith('features.pkl')]
        for p in pooper:
            is_right = "right" in p.lower()
            outname = p[:-4]
            g = Poop.loader("{0}{1}".format(root, p))
            g.imu.data[:, 0] = g.imu.data[:, 0] - offset
            gk = Analysis.find_stride(ikfile=ik,
                                      model_file=osimmodel,
                                      percentage_cpu=-1,
                                      target_markers={'right': {'heel': 'R_Heel', 'toe': 'R_DP1'},
                                                      'left': {'heel': 'L_Heel', 'toe': 'L_DP1'}},
                                      is_debug=False,
                                      show_final=False)
            extracted_gait_parameters = Analysis.process_stride_heel(gk, {'right': 'R_Heel', 'left': 'L_Heel'})
            print("got strides")
            if is_right:
                stride = extracted_gait_parameters['right'][0]
            else:
                stride = extracted_gait_parameters['left'][0]

            fd_raw = g.imu.get_samples(stride['time'].to_list(), as_pandas=True)
            fd = fd_raw.iloc[:, 1:]
            print("filter features")
            efx_filtered = MLOperations.ts_selector(fd, stride['knee_angle_l'])
            print("train regressor to order the features")
            model = MLOperations.train(efx_filtered, stride['knee_angle_l'])
            top_100_feat_dict = MLOperations.select_top_features_from_x(model['feature_importance'], 100)
            MLOperations.export_features_json(top_100_feat_dict['pf_dict'], r"E:\test\{0}_efx_dict.json".format(outname))
            MLOperations.export_features_json(top_100_feat_dict['pfx'], r"E:\test\{0}_efx_dlist.json".format(outname))

            strides = []
            if is_right:
                lefty = extracted_gait_parameters['right']
            else:
                lefty = extracted_gait_parameters['left']
            #
            print("Sort data to strides")
            for i in range(0, len(lefty)):
                stride = lefty[i]
                fd_raw = g.imu.get_samples(stride['time'].to_list(), as_pandas=True)
                fd = fd_raw[top_100_feat_dict['pfx']]
                x = [i for i in range(0, fd.shape[0])]
                fd.insert(0, 'frame', x)
                ydos = Yatsdo(fd)
                dt = fd.shape[0] / 120
                pts = [dt * i for i in range(0, 120)]
                samples = ydos.get_samples(pts)
                strides.append(samples)

            print("Build PCA model")
            data_blocks = [s.ravel() for s in strides]
            pca = PCA()
            pca.setData(np.array(data_blocks).T)

            pca.inc_svd_decompose(None)
            pc = pca.PC

            # outname = "left_foot"
            # outname = "right_foot"
            # outname = "left_thigh"
            # outname = "right_thigh"
            if not os.path.exists("{0}/pc_model/".format(input_dir)):
                os.makedirs("{0}/pc_model/".format(input_dir))
            pc.save("{1}/pc_model/{0}".format(outname, input_dir))
            pc.savemat("{1}/pc_model/{0}.mat".format(outname, input_dir))
            print("Done {0}".format(outname))

def fix_trc():
    wk_dir = "M:/Mocap/"
    out_dir_trc = "E:/meta/Mocap/TRC/"
    out_dir_c3d = "E:/meta/Mocap/TRC/"
    wk = [w for w in os.listdir(wk_dir) if w.startswith('P') and os.path.isdir(wk_dir+w) and 'copy' not in w.lower()]
    ppt_session = {a:'' for a in wk}
    for p in ppt_session:
        dp = [f for f in os.listdir(wk_dir+p) if os.path.isdir("{0}{1}/{2}".format(wk_dir, p, f))]
        ppt_session[p] = dp[0]
    c3d_files = {q: [r for r in os.listdir("{0}{1}/{2}".format(wk_dir, q, ppt_session[q])) if r.endswith('.c3d')] for q
                 in ppt_session}
    trc_files = {q: [r for r in os.listdir("{0}{1}/{2}".format(wk_dir, q, ppt_session[q])) if r.endswith('.trc')] for q
                 in ppt_session}

    for t in trc_files:
        bar = tqdm(range(len(trc_files[t])),
                   desc="Fixing Nexus Exporting Error: " + t,
                   ascii=False, ncols=100, colour="#6e5b5b")
        for s in trc_files[t]:
            try:
                trc_file = "{0}{1}/{2}/{3}".format(wk_dir, t, ppt_session[t], s)
                out = "{0}{1}/{2}".format(out_dir_trc, t, s)
                if not os.path.exists("{0}{1}/".format(out_dir_trc, t)):
                    os.makedirs("{0}{1}/".format(out_dir_trc, t))
                tr = TRC.read(trc_file)
                tr.write(out)
                bar.update(1)

            except ValueError:
                pass
            except IndexError:
                pass
        bar.close()

    for c in c3d_files:
        bar = tqdm(range(len(c3d_files[c])),
                   desc="Copying File to Working Dir: " + c,
                   ascii=False, ncols=100, colour="#6e5b5b")
        for s in c3d_files[c]:
            try:
                c3d_file = "{0}{1}/{2}/{3}".format(wk_dir, c, ppt_session[c], s)
                out = "{0}{1}/{2}".format(out_dir_c3d, c, s)
                if not os.path.exists("{0}{1}/".format(out_dir_c3d, c)):
                    os.makedirs("{0}{1}/".format(out_dir_c3d, c))
                shutil.copy2(c3d_file, out)
                bar.update(1)
            except ValueError:
                pass
            except IndexError:
                pass
        bar.close()

def test():
    print(p)
    for f in straight_line[p]:
        t = TRC.read("{0}{1}/{2}".format(wk, p, f))
        bar = tqdm(range(t.data.shape[1] - 2),
                   desc="Filtering and filling: {0} > {1}".format(p, f),
                   ascii=False, ncols=100, colour="#6e5b5b")
        for d in range(2, t.data.shape[1]):
            fs = int((1 / t.dt) / 6)
            for i in range(0, t.data.shape[0] - fs):
                si = int(fs)
                w = t.data[i:i + si, d]
                non_nans = [j for j in range(0, w.shape[0]) if np.sum(np.isnan(w[j])) == 0]
                dxs = t.data[i:i + si, 1]
                y = w[non_nans]
                x = dxs[non_nans]
                if len(non_nans) < 5:
                    continue
                pf = Polynomial.fit(x, y, 1)
                v = pf(dxs)
                k = np.zeros([2, si])
                k[0, :] = t.data[i:i + si, d]
                k[1, :] = v
                km = np.nanmean(k, axis=0)
                t.data[i:i + si, d] = km
                pass
            dx = Butterworth.butter_low_filter(t.data[:, d], 5, fs)
            t.data[:, d] = dx
            bar.update(1)
        bar.close()
        t.update()
        if not os.path.exists("{0}{1}".format(out_dir, p)):
            os.makedirs("{0}{1}".format(out_dir, p))
        out_file = "{0}{1}/{2}".format(out_dir, p, f)
        t.write(out_file)

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

class MetaMarkerJointSet(Enum):
    hip = {'left': ["L_MedKnee", "L_LatKnee", "L_FibHead", "L_MidShank", "L_MedAnkle", "L_LatAnkle", "L_DP1", "L_MT2",
                    "L_MT5", "L_Heel"],
           'right': ["R_MedKnee", "R_LatKnee", "R_FibHead", "R_MidShank", "R_MedAnkle", "R_LatAnkle", "R_DP1", "R_MT2",
                     "R_MT5", "R_Heel"],
           }

    knee = {'left': ["L_FibHead", "L_MidShank", "L_MedAnkle", "L_LatAnkle", "L_DP1", "L_MT2", "L_MT5", "L_Heel"],
            'right': ["R_FibHead", "R_MidShank", "R_MedAnkle", "R_LatAnkle", "R_DP1", "R_MT2", "R_MT5", "R_Heel"],
            }

    ankle = {'left': ["L_DP1", "L_MT2", "L_MT5", "L_Heel"],
             'right': ["R_DP1", "R_MT2", "R_MT5", "R_Heel"]
             }


class Walker:
    def __init__(self, trc):
        self.joints_upper = {"gh": {"left": None, "right": None},
                             "elbow": {"left": None, "right": None}
                             }
        self.joints_lower = {"hip": {"left": None, "right": None},
                             "knee": {"left": None, "right": None},
                             "ankle": {"left": None, "right": None}}

        self.ref_marker = {"pelvis": None}
        self.ref_trc = trc
        self.keyframe = 0
        self.static_cal(trc)
        # self.setup(trc)

        pass

    def static_cal(self, trc):
        keyframe = 10
        pelvis = np.zeros([3, 5])
        pel = (np.array([trc.marker_set[m].iloc[keyframe, :].to_list() for m in MetaMarkerSet.pelvis.value])).T
        pelvis[:, 0] = np.nanmean(pel, axis=1)
        pelvis[:, 1:] = pel
        pelvis = pelvis - Walker.repeat(pelvis[:, 0], 5)
        pca_pelvis = PCAModel.pca_rotation(pelvis.T)
        pt_pelvis = pca_pelvis.transformation
        ref_pelvis = np.matmul(pt_pelvis, pelvis)
        self.ref_marker["pelvis"] = ref_pelvis

        e0 = np.array(trc.marker_set[MetaMarkerSet.left_upper_leg.value[0]].iloc[keyframe, :].to_list())
        e1 = np.array(trc.marker_set[MetaMarkerSet.left_upper_leg.value[1]].iloc[keyframe, :].to_list())
        k = np.atleast_2d(0.5 * (e0 + e1) - pelvis[:, 0]).T
        k1 = np.matmul(pt_pelvis, k)
        k1_list = np.squeeze(k1).tolist()
        self.joints_lower['knee']['left'] = k1_list

        e0 = np.array(trc.marker_set[MetaMarkerSet.right_upper_leg.value[0]].iloc[keyframe, :].to_list())
        e1 = np.array(trc.marker_set[MetaMarkerSet.right_upper_leg.value[1]].iloc[keyframe, :].to_list())
        k = np.atleast_2d(0.5 * (e0 + e1) - pelvis[:, 0]).T
        k1 = np.matmul(pt_pelvis, k)
        k1_list = np.squeeze(k1).tolist()
        self.joints_lower['knee']['right'] = k1_list

        e0 = np.array(trc.marker_set[MetaMarkerSet.left_lower_leg.value[2]].iloc[keyframe, :].to_list())
        e1 = np.array(trc.marker_set[MetaMarkerSet.left_lower_leg.value[3]].iloc[keyframe, :].to_list())
        k = np.atleast_2d(0.5 * (e0 + e1) - pelvis[:, 0]).T
        k1 = np.matmul(pt_pelvis, k)
        k1_list = np.squeeze(k1).tolist()
        self.joints_lower['ankle']['left'] = k1_list

        e0 = np.array(trc.marker_set[MetaMarkerSet.right_lower_leg.value[2]].iloc[keyframe, :].to_list())
        e1 = np.array(trc.marker_set[MetaMarkerSet.right_lower_leg.value[3]].iloc[keyframe, :].to_list())
        k = np.atleast_2d(0.5 * (e0 + e1) - pelvis[:, 0]).T
        k1 = np.matmul(pt_pelvis, k)
        k1_list = np.squeeze(k1).tolist()
        self.joints_lower['ankle']['right'] = k1_list


    @staticmethod
    def repeat(ar, repeats=4):
        return (np.array([ar for j in range(0, repeats)])).T

    def setup(self, trc):
        self.dynamic_joint(trc)
        pass

    def body_norm(self, v0, v1, v2):
        pv0 = v1 - v0
        pv1 = v2 - v0
        pv0n = pv0 / np.linalg.norm(pv0)
        pv1n = pv1 / np.linalg.norm(pv1)
        pv2n = np.cross(pv0n, pv1n)
        return pv2n

    def track_pelvis(self, trc, outfile=None):
        # this is the root segment and have to be process first
        s0 = self.ref_marker["pelvis"]
        # gaps
        gaps_dic = {}
        for m in MetaMarkerSet.get(MetaMarkerSet.pelvis):
            marker = trc.marker_set[m]
            gaps = []
            new_gaps = 0
            gap_start = -1
            for i in range(0, trc.data.shape[0]):
                n = np.isnan(marker.iloc[i, :])
                if np.sum(n) > 0:
                    if gap_start == -1:
                        gap_start = i-1
                    new_gaps += 1
                else:
                    if new_gaps > 0:
                        gaps.append([gap_start, new_gaps, i])
                        gap_start = -1
                        new_gaps = 0
            gaps_dic[m] = gaps
        pass
        re_run_flag = False
        for m in gaps_dic:
            gaps = gaps_dic[m]
            marker = trc.marker_set[m]
            for g in gaps:
                if g[0] > 0:
                    gap = []
                    frame0 = marker.iloc[g[0], :].to_list()
                    frame1 = marker.iloc[g[-1], :].to_list()
                    for i in range(0, 3):
                        y = [frame0[i], frame1[i]]
                        x = [0, g[1]+1]
                        xt = [j for j in range(0, g[1]+2)]
                        coefficients = np.polyfit(x, y, 1)
                        yt = np.polyval(coefficients, xt)
                        gap.append(yt[1])
                    gap = np.array(gap)
                    gap_x = [xc for xc in range(g[0]+1, g[-1]-1)]
                    possi = []
                    for n in trc.marker_set:
                        marker_name = n
                        possible = np.array(trc.marker_set[n].iloc[g[0]+1, :].to_list())
                        if np.sum(np.isnan(possible)) > 0:
                            continue

                        v = np.linalg.norm(gap-possible)
                        if len(possi) == 0:
                            possi=[n, v, possible]
                        else:
                            if v < possi[1]:
                                possi = [n, v, possible]
                        pass
                    if possi[0] in '*':
                        re_run_flag = True
                    else:
                        for x in gap_x:
                            trc.marker_set[m].iloc[x, :] = possi[2]
                    pass

        for i in range(0, trc.data.shape[0]):
            pelvis = np.zeros([3, 5])
            marker_list = MetaMarkerSet.get(MetaMarkerSet.pelvis)
            pel = (np.array([trc.marker_set[m].iloc[i, :].to_list() for m in marker_list])).T
            translate = np.nanmean(pel, axis=1)
            pelvis[:, 0] = translate
            pelvis[:, 1:] = pel

            pelvis_prev = np.zeros([3, 5])
            pv2n0 = np.array([1, 0, 0])
            if i > 0:
                pel_prev = (np.array([trc.marker_set[m].iloc[i - 1, :].to_list() for m in marker_list])).T
                translate_prev = np.nanmean(pel_prev, axis=1)
                pelvis_prev[:, 0] = translate_prev
                pelvis_prev[:, 1:] = pel_prev
                pv2n0 = self.body_norm(pelvis_prev[:, 0], pelvis_prev[:, 2], pelvis_prev[:, 1])
            n = np.isnan(pelvis[0, :])
            k = [i for i in range(1, n.shape[0]) if not n[i]]

            if len(k) >=3:
                # km = list(permutations(k, len(k)))
                pmat = []
                # for j in km:
                #     s1 = s0[:, j]
                #     s1a = np.ones([4, s0.shape[1]])
                #     s1a[:3, :] = s0
                #     p1 = pelvis[:, j]
                #     rt3 = Cloud.transform_between_3x3_points_sets(s1, p1)
                #     p1fe = np.matmul(rt3, s1a)
                #     pmat.append(p1fe)

                mk = [m0 for m0 in trc.marker_names if '*' in m0]
                possibles0 = {}
                for kk in mk:
                    mt = np.array(trc.marker_set[kk].iloc[i, :].to_list())
                    if np.sum(np.isnan(mt)) == 0:
                        possibles0[kk] = mt
                possibles0_keys = [kk for kk in possibles0]
                pelvis_2 = np.zeros([3, 4+len(possibles0_keys)])
                marker_list = MetaMarkerSet.get(MetaMarkerSet.pelvis)
                pel = (np.array([trc.marker_set[m].iloc[i, :].to_list() for m in marker_list])).T
                pelvis_2[:, :4] = pel
                for p1 in range(0, len(possibles0_keys)):
                    pelvis_2[:, 4 + p1] = np.squeeze(possibles0[possibles0_keys[p1]])
                    marker_list.append(possibles0_keys[p1])
                n = np.isnan(pelvis_2[0, :])
                k = [i for i in range(0, n.shape[0]) if not n[i]]
                segment_marker_no = len(MetaMarkerSet.pelvis.value)
                km = list(permutations(k, segment_marker_no))
                for j in km:
                    s1 = s0[:, 1:]
                    s1a = np.ones([4, s0.shape[1]])
                    s1a[:3, :] = s0
                    p1 = pelvis_2[:, j]
                    rt3 = Cloud.transform_between_3x3_points_sets(s1, p1)
                    p1fe = np.matmul(rt3, s1a)
                    pmat.append(p1fe)
                p1f = pmat[0]
                if i > 0:
                    if i==434:
                        pass
                    diffs = []
                    per_idx = 0
                    for p in pmat:
                        diff = p[:3, :] - pelvis_prev
                        pv2n1 = self.body_norm(p[:3, 0], p[:3, 2], p[:3, 1])
                        norm_diff = np.linalg.norm(pv2n1 - pv2n0)
                        norms = []
                        for q in range(0, 5):
                            nor = np.linalg.norm(diff[:, q])
                            norms.append(nor)
                        ret = [np.sum(norms), per_idx, norm_diff]
                        per_idx += 1
                        for rt in range(0, 3):
                            ret.append(pv2n1[rt])
                        diffs.append(ret)
                        pass
                    diffs = np.array(diffs)
                    sorted_indices = np.argsort(diffs[:, 2])
                    sorted_array = diffs[sorted_indices]
                    p1f = pmat[int(sorted_array[0, 1])]
                    pass
                idx = 1
                marker_list = MetaMarkerSet.get(MetaMarkerSet.pelvis)
                marker_list.insert(0, '')
                for m in range(1, len(marker_list)):
                    trc.marker_set[marker_list[m]].iloc[i, :] = p1f[:3,m]
                pass

            pass
        marker_list = MetaMarkerSet.get(MetaMarkerSet.pelvis)
        # for m in marker_list:
        #     for i in range(0, 3):
        #         y = trc.marker_set[m].iloc[:, i]
        #         trc.marker_set[m].iloc[:, i] = Butterworth.butter_low_filter(y, 5, 100)
        trc.update_from_markerset()
        #trc.headers['Units'] = 'mm'
        if outfile is not None:
            trc.write(outfile)
        pass


    def dynamic_joint(self, trc):
        i = 0
        torso = np.zeros([3, 4])
        tor = (np.array([trc.marker_set[m].iloc[i, :].to_list() for m in MetaMarkerSet.torso.value])).T
        torso[:, 0] = np.nanmean(tor, axis=1)
        torso[:, 1:] = tor
        torso = torso - Walker.repeat(torso[:, 0])
        pca_torso = PCAModel.pca_rotation(torso.T)
        ref_torso = (pca_torso.transformed_data).T
        #######################################################################################################
        # gh joints
        self.joints_upper['gh'] = self.gh_joint(trc, ref_torso)
        torso = np.zeros([3, 4])
        tor = (np.array([trc.marker_set[m].iloc[8, :].to_list() for m in MetaMarkerSet.torso.value])).T
        torso[:, 0] = np.nanmean(tor, axis=1)
        torso[:, 1:] = tor
        rt = Cloud.rigid_body_transform(torso, ref_torso)
        l_elbow = np.mean([trc.marker_set[c].iloc[8, :] for c in MetaMarkerSet.left_upper_arm.value], axis=0)
        xL = np.ones([4, 1])
        xL[:3, :] = (np.atleast_2d(l_elbow)).T
        r_elbow = np.mean([trc.marker_set[c].iloc[8, :] for c in MetaMarkerSet.right_upper_arm.value], axis=0)
        xR = np.ones([4, 1])
        xR[:3, :] = (np.atleast_2d(r_elbow)).T
        #######################################################################################################
        # elbow centres
        self.joints_upper['elbow']['left'] = np.squeeze((np.matmul(rt, xL))[:3, :])
        self.joints_upper['elbow']['right'] = np.squeeze((np.matmul(rt, xR))[:3, :].T)
        ########################################################################################################
        # Hip Joint Centre
        self.hip_joint(trc)
        JSONSUtl.write_json("./test.json", self.joints_lower)

    def gh_joint(self, trc, ref):
        left_shoulder_list = []
        right_shoulder_list = []
        for i in range(1, trc.data.shape[0]):
            torso = np.zeros([3, 4])
            tor = (np.array([trc.marker_set[m].iloc[i, :].to_list() for m in MetaMarkerSet.torso.value])).T
            torso[:, 0] = np.nanmean(tor, axis=1)
            torso[:, 1:] = tor
            rt = Cloud.rigid_body_transform(torso, ref)
            ones = np.ones([4, 4])
            ones[:3, :] = torso
            t1 = np.matmul(rt, ones)

            lup = {}
            for c in MetaMarkerSet.left_upper_arm.value:
                x = np.ones([4, 1])
                mk = np.atleast_2d(trc.marker_set[c].iloc[i, :].to_list())
                x[:3, 0] = mk
                x1 = np.matmul(rt, x)
                lup[c] = x1[:3, :].T
            L_elbow = (lup[MetaMarkerSet.left_upper_arm.value[0]] + lup[MetaMarkerSet.left_upper_arm.value[1]]) / 2
            left_shoulder_list.append(np.squeeze(L_elbow))

            rup = {}
            for c in MetaMarkerSet.right_upper_arm.value:
                x = np.ones([4, 1])
                mk = np.atleast_2d(trc.marker_set[c].iloc[i, :].to_list())
                x[:3, 0] = mk
                x1 = np.matmul(rt, x)
                rup[c] = x1[:3, :].T
            r_elbow = (rup[MetaMarkerSet.right_upper_arm.value[0]] + rup[
                MetaMarkerSet.right_upper_arm.value[1]]) / 2
            right_shoulder_list.append(np.squeeze(r_elbow))
            pass
        lefty = np.array([n for n in left_shoulder_list if np.sum(np.isnan(n)) == 0])
        rc = np.squeeze(Cloud.sphere_fit(lefty))
        left_shoulder = rc

        righty = np.array([n for n in right_shoulder_list if np.sum(np.isnan(n)) == 0])
        rc = np.squeeze(Cloud.sphere_fit(righty))
        right_shoulder = rc
        return {'left': left_shoulder, 'right': right_shoulder}

    def hip_joint(self, trc):
        ref = self.ref_marker["pelvis"]

        def filter_marker(marker_set):
            markers = {}
            for c in marker_set:
                dx = trc.marker_set[c].to_numpy()
                xt = np.array([trc.data[i, 1] for i in range(0, dx.shape[0])])
                x = np.array([trc.data[i, 1] for i in range(0, dx.shape[0]) if np.sum(np.isnan(dx[i])) == 0])
                xi = np.array([i for i in range(0, dx.shape[0]) if np.sum(np.isnan(dx[i])) == 0])
                dxc = np.array([dx[i] for i in range(0, dx.shape[0]) if np.sum(np.isnan(dx[i])) == 0])
                dxd = np.zeros(dx.shape)
                for i in range(0, 3):
                    p = interpolate.InterpolatedUnivariateSpline(x, dxc[:, i])
                    v0 = p(xt)
                    v1 = Butterworth.butter_low_filter(v0, 6, 100)
                    dxd[:, i] = v1
                markers[c] = {'i': [j for j in range(xi[0], xi[-1])], 'x': xt[xi[0]:xi[-1]],
                                   'y': dxd[xi[0]:xi[-1], :]}
            return markers

        left_markers = filter_marker(MetaMarkerSet.left_upper_leg.value)
        right_markers = filter_marker(MetaMarkerSet.right_upper_leg.value)

        def rc_cal(markers):
            left_knee_list = []
            mc = [c for c in markers]
            ts = mc[0]
            if markers[mc[1]]['i'][0] > markers[ts]['i'][0]:
                ts = markers[mc[1]]['i'][0]
            idx_list = markers[ts]['i']
            for i in idx_list:
                pelvis = np.zeros([3, 5])
                pel = (np.array([trc.marker_set[m].iloc[i, :].to_list() for m in MetaMarkerSet.pelvis.value])).T
                if np.sum(np.isnan(pel)) > 0:
                    continue
                translate = np.nanmean(pel, axis=1)
                pelvis[:, 0] = translate
                pelvis[:, 1:] = pel
                pelvis = pelvis - Walker.repeat(pelvis[:, 0], 5)
                rt = Cloud.transform_between_3x3_points_sets(ref, pelvis)[:3, :3]

                mk2 = {}
                for c in markers:
                    mx = np.empty(trc.marker_set[c].shape)
                    mx[:] = np.nan
                    for z in range(0, markers[c]['y'].shape[0]):
                        mx[markers[c]['i'][z], :] = markers[c]['y'][z, :]
                    mk2[c] = mx

                lk0 = np.transpose(np.array([mk2[c][i, :] for c in mk2]))
                if np.sum(np.isnan(lk0)) > 0:
                    continue
                lk1 = lk0-Walker.repeat(translate, 2)
                x1 = np.matmul(rt, lk1)
                L_knee = np.squeeze(np.nanmean(x1, axis=1).tolist())
                left_knee_list.append(np.squeeze(L_knee))
            lefty = np.squeeze(left_knee_list)
            # plt.figure()
            # plt.plot(lefty)
            # plt.show()
            # plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.scatter(lefty[:, 0], lefty[:, 1], lefty[:, 2])
            # plt.show()
            return np.squeeze(Cloud.sphere_fit(lefty))
        left_hip = rc_cal(left_markers)
        right_hip = rc_cal(right_markers)

        self.joints_lower["hip"]["left"] = left_hip.tolist()
        self.joints_lower["hip"]["right"] = right_hip.tolist()
        return None


    def move(self):
        # pelvis lock
        ### check markers
        ### if a marker missing if there is atleast 3 markers, reconstruct using current frame markers
        ### if 2 or more use spline fit requires window, project forward/ back 20.
        # fabik using joint location
        ### hip joint move with pelvis
        ### hip is anchor
        ### knee is 1 Dof
        ### ankle is 2 Dof

        pass


if __name__ == '__main__':


    s = StorageIO.load("E:/meta/Mocap/mot/P025/Straight_norm_tracked_a.mot")
    pass
    # Poop.normal_model_v1()
    # fix_trc()
    #IK.run_from_c3d_1(wk_dir,trial, template, model)

    wk = "E:/meta/Mocap/Optical/"
    out_dir = "E:/meta/Mocap/Filter/"
    participants = {p:[f for f in os.listdir("{0}{1}/".format(wk, p)) if f.endswith(".c3d")] for p in os.listdir(wk)}
    #straight_line = {p:[f for f in participants[p] if "straight" in f.lower() and 'norm' in f.lower()] for p in participants}
    straight_line = {p: [f for f in participants[p] if "straight" in f.lower() and 'vr' in f.lower()] for p in
                     participants}
    w = None
    for p in straight_line:
        if p == 'P025':
            trc = TRC.create_from_c3d("{0}{1}/{2}".format(wk, p, "static 01.c3d"), fill_data=False) # this needs to be changed
            w = Walker(trc)
            for f in straight_line[p]:
                trc = TRC.create_from_c3d("{0}{1}/{2}".format(wk, p, f), fill_data=False)
                trial = f[:-4]
                outfile = "{0}{1}/{2}_tracked.trc".format(wk, p, trial)
                w.setup(trc)
                w.track_pelvis(trc, outfile)
                pass
            break

    # for p in straight_line:
    #         for f in straight_line[p]:
    #             trc = TRC.create_from_c3d("{0}{1}/{2}".format(wk, p, f), fill_data=False)
    #             w.setup(trc)
    #             pass
    #
    #         pass
    pass

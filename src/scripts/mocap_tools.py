import os
from ptb.util.gait.helpers import OsimHelper
from ptb.util.data import MocapDO, TRC
from ptb.util.math.transformation import Cloud, Quaternion

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from opensim.common import SimmSpline
from opensim import TransformAxis, SpatialTransform
from enum import Enum
from tqdm import tqdm

import copy
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool

cost_ret = []
def eul_dist(x, y):
    d = x - y
    return np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])

def cost(v, model, mocap):
    p = [c for c in model.state_variable_names_processed if 'N_A' not in c]
    super_error = [1.0]
    for vp in range(0, 3):
        if v[vp] > 180.0 or v[vp] < -180.0 or np.sum(np.isnan(v[vp])) > 0:
            super_error.append(1e10)
    for vp in range(6, v.shape[0]):
        if v[vp] > 180.0 or v[vp] < -180.0 or np.sum(np.isnan(v[vp])) > 0:
            super_error.append(1e10)
    if np.sum(super_error)>1.0:
        return np.sum(super_error)
    plea = pd.Series(data=v, index=p)
    model.set_joints(plea)
    k = copy.deepcopy(model.markerset)
    knp = k.to_numpy()
    mnp = mocap
    if isinstance(mnp, pd.DataFrame):
        mnp = mocap.to_numpy()

    error = []
    error0 = []
    for i in range(0, k.shape[1]):
        g = 1000*eul_dist(knp[:, i], mnp[:, i])
        error.append(np.abs(g))
        error0.append(g*g)

    error.sort()
    ae = float(np.nanmean(error))
    se = float(np.nanstd(error))
    ratio = ((se / ae) * 100.0)
    m1a = float(np.sqrt(np.nanmean(error0)))
    m1s = float(np.sqrt(np.nanstd(error0)))
    me = 1000* np.nanmax(error)
    ratio1 = (m1s/m1a)*100

    error1 = []
    error2 = []
    baseme = []

    for i in range(0, len(error)):
        e = error[i]
        b = np.power(10, (i/10))
        c = (b * 1000 * e)
        baseme.append(b*1000)
        error1.append(c)
        error2.append(c*c)

    m2 = float(np.sqrt(np.nanmean(error2)))

    ret = (ratio + m2 + ratio1 + m1a + me)/ np.nansum(baseme)
    #print("ae, {0:.5f}, ret, {1:.5f}, sep, {2:.5f}, me, {3:.5f}".format(m1a, ret, se, np.max(error)))
    cost_ret.append(ret)
    return ret


def marker_track_test():
    OsimHelper.add_geo_search_path(r"C:\OpenSim 4.5\Geometry")
    model_dir = "M:/temp/"
    mt = [m for m in os.listdir(model_dir) if m.lower().startswith("p")]
    for m in mt:
        particpant = m[:m.rindex(".")]
        mf = [s for s in os.listdir("M:/Mocap/{0}/".format(particpant)) if 'session' in s.lower()]
        opens_model = OsimHelper("{0}{1}.osim".format(model_dir, particpant))
        p = [c for c in opens_model.state_variable_names_processed if 'N_A' not in c]
        keys = [k for k in opens_model.markerset]
        trials = [t for t in os.listdir("M:/Mocap/{0}/{1}".format(particpant, mf[0])) if t.endswith("c3d")]
        trials.sort()
        static_trials = [t for t in trials if "cal" in t.lower() or 'static' in t.lower()]
        k0 = copy.deepcopy(opens_model.markerset)
        mocap_data = MocapDO.create_from_c3d("M:/Mocap/{0}/{1}/{2}".format(particpant, mf[0], static_trials[0]))
        mocap_data.markers.z_up_to_y_up()
        mset = mocap_data.marker_set
        unit = 1
        if mocap_data.markers.headers['Units'] == 'mm':
            unit = 0.001
        frames = mocap_data.markers.data.shape[0]
        frame = int(np.round(frames / 2.0, 0))

        tf = np.array([(unit * mset[t].iloc[frame, :]).to_list() for t in keys])
        n0 = k0.to_numpy()
        n1 = tf.T
        where_nans = np.isnan(n1[0, :])
        n0a = np.zeros([3, int(n0.shape[1] - np.sum(where_nans))])
        n1a = np.zeros([3, int(n0.shape[1] - np.sum(where_nans))])
        idx = 0
        for i in range(0, n0.shape[1]):
            if not where_nans[i]:
                n0a[:, idx] = n0[:, i]
                n1a[:, idx] = n1[:, i]
                idx += 1
        ct = Cloud.rigid_body_transform(n0a, n1a)
        r = Rotation.from_matrix(ct[:3, :3])
        rx = r.as_euler('zxy', degrees=True)
        x0 = np.array([0.0 for i in range(0, len(p))])
        x0[:3] = rx
        x0[3:6] = ct[:3, 3]
        opens_model.set_joints(x0)
        result = minimize(cost, x0, args=(opens_model, n1,), method='Powell')
        x1 = result.x
        print(cost(x1, opens_model, n1))
        opens_model.set_joints(x1)
        n2 = (copy.deepcopy(opens_model.markerset)).to_numpy()
        col = [c for c in opens_model.markerset]
        nx = np.zeros(n2.shape)
        for i in range(0, n2.shape[1]):
            if not where_nans[i]:
                nx[:, i] = 0.1 * n2[:, i] + 0.9 * n1[:, 1]
            else:
                nx[:, i] = n2[:, i]
            pass

        for c in opens_model.markerset:
            cur_marker = nx[:, col.index(c)] * 1 / unit
            cur_frame_trc = mocap_data.marker_set[c].iloc[frame, :]
            mocap_data.marker_set[c].iloc[frame, :] = nx[:, col.index(c)]
            pass
        mocap_data.markers.update_from_markerset()
        pass


class MetaModel(Enum):
    torso = ["Sternum", "R_Acromion", "L_Acromion"]
    head = ["Head"]
    r_leg = ["R_FibHead", "R_LatKnee", "R_MedKnee", "R_MidShank"]
    l_leg = ["L_FibHead", "L_LatKnee", "L_MedKnee", "L_MidShank"]
    r_foot = ["R_Heel", "R_MT5", "R_MT2", "R_DP1", "R_MedAnkle", "R_LatAnkle"]
    l_foot = ["L_Heel", "L_MT5", "L_MT2", "L_DP1", "L_MedAnkle", "L_LatAnkle"]
    r_arm = ["R_Lat_HumEpicondyle", "R_Med_HumEpicondyle", "R_Radius", "R_Ulna"]
    l_arm = ["L_Lat_HumEpicondyle", "L_Med_HumEpicondyle", "L_Radius", "L_Ulna"]
    pelvis = ["RPSIS", "LPSIS", "RASIS", "LASIS"]


    def center(self, markerset: dict):
        matrix = np.zeros([3, len(self.value)])
        idx = 0
        for v in self.value:
            matrix[:, idx] = markerset[v]
            idx += 1
        return np.nanmean(matrix, axis=1)

    def ref(self, markerset: dict):
        matrix = np.zeros([3, len(self.value)])
        idx = 0
        for v0 in self.value:
            matrix[:, idx] = markerset[v0]
            idx += 1
        c0 = np.nanmean(matrix, axis=1)
        m0 = np.array([matrix[:, n]- c0 for n in range(0, len(self.value))])
        return m0

class MocapTools:
    @staticmethod
    def cleanup_nexus_trc(file=r"M:\Mocap\P016\New Session\Static calib 02.trc"):
        if "re-exported" in file:
            return
        trsp = os.path.split(file)
        trname = trsp[1][:trsp[1].rindex('.')]
        tr = TRC.read(file)
        out_name = '{0}/{1}_re-exported.trc'.format(trsp[0], trname)
        tr.write(out_name)

    @staticmethod
    def quaternion_bridge(start, end, p0, frames):
        bridge = end - start
        p1 = (np.array([frames[end][c] for c in frames[end]])).T
        t = Cloud.transform_between_3x3_points_sets(p0, p1)
        r = t[:3, :3]
        d = t[:3, 3]
        ix = np.eye(3)
        k0 = Rotation.from_matrix(ix)
        q0 = k0.as_quat(scalar_first=True)
        k = Rotation.from_matrix(r)
        q = k.as_quat(scalar_first=True)
        qa = Quaternion(q0)
        qb = Quaternion(q)
        qc = []
        for bg in range(1, bridge):
            ratio = bg / bridge
            qd = Quaternion.slerp(qa, qb, ratio)
            rd = Rotation.from_quat(qd.to_array(), scalar_first=True)
            rm = rd.as_matrix()
            tm = d * (ratio)
            t1 = np.eye(4)
            t1[:3, :3] = rm
            t1[:3, 3] = tm
            p0a = np.vstack([p0, [1 for i in range(0, p0.shape[1])]])
            p01a = np.matmul(t1, p0a)
            for c in range(0, len(MetaModel.torso.value)):
                w = MetaModel.torso.value[c]
                idx = start + bg
                frames[idx][w] = p01a[:3, c]
                pass
            qc.append(qd)

    @staticmethod
    def cleanup_unlabel(mocap_data, unlabel='*'):
        ke = [f for f in mocap_data.marker_set if f.startswith(unlabel)]
        for v in ke:
            mocap_data.marker_set.pop(v)

    @staticmethod
    def pandas2npy(x):
        return x.to_numpy()

    @staticmethod
    def static_trial_cleaner(file, do_z2y=True):
        px = os.path.split(file)
        pxname = px[1][:px[1].rindex('.')]
        mocap_data = MocapDO.create_from_c3d(file)
        markers_in_frame = {}
        for v in MetaModel.torso.value:
            e = [f for f in mocap_data.marker_set if v.lower() in f.lower()][0]
            markers_in_frame[v] = mocap_data.marker_set[e].to_numpy()[1, :]
        frames = []
        for i in range(0, mocap_data.markers.data.shape[0]):
            markers_in_frame = {}
            boo = False
            nana = []
            for v in MetaModel.torso.value:
                e = [f for f in mocap_data.marker_set if v.lower() in f.lower()][0]
                point = mocap_data.marker_set[e].to_numpy()[i, :]
                if np.sum(np.isnan(point)) > 0:
                    boo = True
                    nana.append(e)
                else:
                    markers_in_frame[v] = mocap_data.marker_set[e].to_numpy()[i, :]
            if boo:
                frames.append({})
                pass
            else:
                frames.append(markers_in_frame)
            pass
        for f in range(0, len(frames)):
            nana = len(frames[f].keys())
            if nana == 0:
                start = f - 1
                p0 = (np.array([frames[start][c] for c in frames[start]])).T
                end = -1
                for j in range(f, len(frames)):
                    nana = len(frames[j].keys())
                    if nana > 0:
                        print()
                        end = j
                        break
                if end > -1:
                    MocapTools.quaternion_bridge(start, end, p0, frames)

        for i in range(0, mocap_data.markers.data.shape[0]):
            for v in MetaModel.torso.value:
                e = [f for f in mocap_data.marker_set if v.lower() in f.lower()][0]
                mocap_data.marker_set[e].iloc[i, :] = frames[i][v]
        for mk in MetaModel:
            for v in mk.value:
                e = [f for f in mocap_data.marker_set if v.lower() in f.lower()][0]
                mocap_data.markers.marker_set[v] = mocap_data.marker_set.pop(e)
        ke = [f for f in mocap_data.marker_set if f.startswith('*')]
        me = [f for f in mocap_data.marker_set if not f.startswith('*')]
        for mf in me:
            for f in range(0, mocap_data.marker_set[mf].shape[0]):
                nana = np.sum(np.isnan(mocap_data.marker_set[mf].iloc[f, :]))
                if nana > 0:
                    possible_m = None
                    diff = np.inf
                    poss_cur = None
                    prev = MocapTools.pandas2npy(mocap_data.marker_set[mf].iloc[f - 1, :])
                    for u in ke:
                        nana_u = np.sum(np.isnan(mocap_data.marker_set[u].iloc[f, :]))
                        if nana_u == 0:
                            n = MocapTools.pandas2npy(mocap_data.marker_set[u].iloc[f, :])
                            df = np.linalg.norm(n - prev)
                            if df < diff:
                                diff = df
                                possible_m = u
                                poss_cur = n
                    mocap_data.marker_set[mf].iloc[f, :] = poss_cur
                    pass

        MocapTools.cleanup_unlabel(mocap_data)
        mocap_data.markers.update_from_markerset()
        if do_z2y:
            mocap_data.markers.z_up_to_y_up()

        outf = "{0}/{1}.trc".format(px[0], pxname)
        mocap_data.markers.write(outf)

def worker(tt):
    try:
        MocapTools.cleanup_nexus_trc(tt)
    except ValueError:
        pass
    except IndexError:
        pass

if __name__ == '__main__':
    work_dir = "M:/Mocap/"
    particip =  [p for p in os.listdir(work_dir) if os.path.isdir("{0}{1}".format(work_dir, p)) and p.lower().startswith('p')]
    session = {p: [d for d in os.listdir("{0}{1}".format(work_dir, p)) if os.path.isdir("{0}{1}/{2}".format(work_dir, p, d))] for p in particip}
    bar = tqdm(range(len(particip)),
               desc="scan for trc",
               ascii=False, ncols=100, colour="#6e5b5b")
    trials = {}
    for p in particip:
        tr = []
        for s in session[p]:
            for t in os.listdir("{0}{1}/{2}".format(work_dir, p, s)):
                if t.lower().endswith('trc'):
                    tr.append("{0}{1}/{2}/{3}".format(work_dir, p, s, t))
        trials[p] = tr
        bar.update(1)
    bar.close()

    bar = tqdm(range(len(particip)),
               desc="re-export trc",
               ascii=False, ncols=100, colour="#6e5b5b")
    poo = Pool(8)
    for p in trials:
        work = trials[p]
        ret = poo.map(worker, work)
        bar.update(1)
    bar.close()

    pass
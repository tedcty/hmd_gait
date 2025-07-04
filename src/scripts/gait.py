
import pandas as pd
from ptb.util.gait.helpers import OsimHelper
from ptb.util.data import MocapDO
from ptb.util.math.transformation import Cloud

import numpy as np
import copy
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import os

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

if __name__ == '__main__':
    model_dir = "M:/temp/"
    mt = [m for m in os.listdir(model_dir) if m.lower().startswith("p")]
    mt = ['P009.osim']
    for m in mt:
        particpant = m[:m.rindex(".")]
        mf = [s for s in os.listdir("M:/Mocap/{0}/".format(particpant)) if 'session' in s.lower()]
        opens_model = OsimHelper("{0}{1}.osim".format(model_dir, particpant))
        k0 = copy.deepcopy(opens_model.markerset)

        trials = [t for t in os.listdir("M:/Mocap/{0}/{1}".format(particpant, mf[0])) if t.endswith("c3d") and t =='Free normal 1.c3d']

        p = [c for c in opens_model.state_variable_names_processed if 'N_A' not in c]
        keys = [k for k in opens_model.markerset]
        for t in trials:
            mocap_data = MocapDO.create_from_c3d("M:/Mocap/{0}/{1}/{2}".format(particpant, mf[0], t))
            mset = mocap_data.markers.marker_set
            unit = 1
            if mocap_data.markers.headers['Units'] == 'mm':
                unit = 0.001
            frames = mocap_data.markers.data.shape[0]
            frame = 211
            tf = np.array([(unit * mset[t].iloc[frame, :]).to_list() for t in keys])
            n0 = k0.to_numpy()
            n1 = tf.T
            where_nans = np.isnan(n1[0, :])
            n0a = np.zeros([3, int(n0.shape[1]-np.sum(where_nans))])
            n1a = np.zeros([3, int(n0.shape[1] - np.sum(where_nans))])
            idx = 0
            for i in range(0, n0.shape[1]):
                if not where_nans[i]:
                    n0a[:, idx] = n0[:, i]
                    n1a[:, idx] = n1[:, i]
                    idx+= 1
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
                    nx[:, i] = 0.1*n2[:, i]+ 0.9*n1[:, 1]
                else:
                    nx[:, i] = n2[:, i]
                pass

            for c in opens_model.markerset:
                cur_marker = nx[:, col.index(c)]*1000
                cur_frame_trc = mocap_data.marker_set[c].iloc[frame, :]
                mocap_data.marker_set[c].iloc[frame, :] = nx[:, col.index(c)]
                pass
            pass


    opens_model = OsimHelper("M:/temp/P009.osim")
    k0 = copy.deepcopy(opens_model.markerset)
    keys = [k for k in k0.columns]
    mocap_data = MocapDO.create_from_c3d("M:/Mocap/P009/New Session/Free normal 1.c3d")
    mset = mocap_data.markers.marker_set
    unit = 1
    if mocap_data.markers.headers['Units'] == 'mm':
        unit = 0.001
    frame = mocap_data.markers.data.shape[0]
    x0 =None

    for j in range(0, frame):
        cost_ret = []
        print(j)
        if x0 is not None and np.sum(np.isnan(x0)):
            pass
        tf = np.array([(unit * mset[t].iloc[j, :]).to_list() for t in keys])
        p = [c for c in opens_model.state_variable_names_processed if 'N_A' not in c]
        # pl0 = np.array([1.34122165, -5.28454454, -88.606435, 0.28949265, 1.10793158, -2.2650449, -8.19955895, 0.00799601, -6.35545273, 2.45149332, 0.04278663, 9.79497742, 9.99673188, 0, 19.87424033, -6.26259015, -9.35120964, 50.8312105, 0.88717199, 1.22068266, -1.55341301, 0, -15.9077124, 0.5362555, 3.22569492, 1.34457939, -8.05451137, 6.25662265, 30.64010864, 87.57262748, 0, 5.00000014, -4.33143493, -7.22046942, -2.3899443, 32.00328326, 88.41433085, 0, 5.00000014])
        # if x0 is None:
        #     pl = np.array([2.60547836,-3.49346285,-84.07186503,0.26597638,1.08609175,-4.65127019,-5.05819393,-4.0998606,-9.57972802,1.6383902,0.0285953,8.91977696,10.89634743,0,-4.92187866,-0.28560929,-4.61502505,0.59306057,0.01035086,5.98996714,16.15712293,0,-9.76980297,-0.59067699,-3.36010955,-1.28336317,-5.8375015,7.99760452,25.94473379,85.39498594,0,3.11874812,-5.73698062,-7.22258273,7.24965302,34.48362305,86.73771181,0,3.11874812])
        #     ple = pd.Series(data=pl, index=p)
        # else:
        #     ple = x0
        # opens_model.set_joints(ple) # current assumes angles in degrees
        # k1 = copy.deepcopy(opens_model.markerset)
        # k1 = k1.to_numpy()
        met = 'Powell'
        if x0 is not None:
            #met = 'BFGS'
            met = 'L-BFGS-B'
        k1 = tf.T

        n0 = k0.to_numpy()
        n1 = k1
        try:

            if x0 is None:
                ct = Cloud.rigid_body_transform(n0, n1)
                r = Rotation.from_matrix(ct[:3, :3])
                rx = r.as_euler('zxy', degrees=True)
                x0 = np.array([0.0 for i in range(0, len(p))])
                x0[:3] = rx
                x0[3:6] = ct[:3, 3]
            opens_model.set_joints(x0)
            nx = int(np.round((len(x0)*1000)/3.0, 0))
            print (met)
            result = minimize(cost, x0, args=(opens_model, k1,), method=met)
            x1 = result.x
            print(cost(x1, opens_model, k1))
            opens_model.set_joints(x1)
            k2 = copy.deepcopy(opens_model.markerset)
            k2 = k2.to_numpy()
            error = []
            error1 = []
            for i in range(0, k1.shape[1]):
                d = eul_dist(k2[:, i], k1[:, i])
                error.append(d)
                error1.append(d*d)
            error = np.array(error)
            error1 = np.sqrt(np.nanmean(error1))
            print(len(cost_ret))
            result_ret = np.array(cost_ret)
            s = pd.Series(data=result_ret)
            s.to_csv("C:/Users/tyeu008/Downloads/iter_cost_{0}.csv".format(j))
            s = pd.Series(data=error)
            s.to_csv("C:/Users/tyeu008/Downloads/marker_error_{0}.csv".format(j))
            if np.sum(np.isnan(x1)) == 0:
                x0 = copy.deepcopy(x1)
                s = pd.Series(data=x1, index=p)
                s.to_csv("C:/Users/tyeu008/Downloads/angles_{0}.csv".format(j))
            else:
                result = minimize(cost, x0, args=(opens_model, k1,), method='BFGS')
                x1 = result.x
                print(cost(x1, opens_model, k1))
                opens_model.set_joints(x1)
                k2 = copy.deepcopy(opens_model.markerset)
                k2 = k2.to_numpy()
                error = []
                error1 = []
                for i in range(0, k1.shape[1]):
                    d = eul_dist(k2[:, i], k1[:, i])
                    error.append(d)
                    error1.append(d * d)
                error = np.array(error)
                error1 = np.sqrt(np.nanmean(error1))
                result_ret = np.array(cost_ret)
                s = pd.Series(data=result_ret)
                s.to_csv("C:/Users/tyeu008/Downloads/iter_cost_{0}.csv".format(j))
                s = pd.Series(data=error)
                s.to_csv("C:/Users/tyeu008/Downloads/marker_error_{0}.csv".format(j))
                if np.sum(np.isnan(x1)) == 0:
                    x0 = copy.deepcopy(x1)
                else:
                    pass
            print(nx)
        except np.linalg.LinAlgError:
            print(str(j)+"_error")

            break
    pass

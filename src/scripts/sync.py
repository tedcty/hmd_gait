import os

from ptb.util.data import TRC, Yatsdo
from ptb.util.math.filters import Butterworth
from scipy import signal
from scipy import interpolate

from tqdm import tqdm

import numpy as np
import pandas as pd
import warnings
from src.util.meta import Param
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from ptb.util.io.helper import drive
from ptb.util.data import JSONSUtl


def initialise(work_dir = "M:/Mocap/", ext='trc'):
    particip = [p for p in os.listdir(work_dir) if
                os.path.isdir("{0}{1}".format(work_dir, p)) and p.lower().startswith('p')]
    particip.sort()
    session = {
        p: [d for d in os.listdir("{0}{1}".format(work_dir, p)) if os.path.isdir("{0}{1}/{2}".format(work_dir, p, d))]
        for p in particip}
    bar = tqdm(range(len(particip)),
               desc="scan for {0}".format(ext),
               ascii=False, ncols=100, colour="#6e5b5b")
    trials = {}
    for p in particip:
        tr = []
        for s in session[p]:
            for t in os.listdir("{0}{1}/{2}".format(work_dir, p, s)):
                if t.lower().endswith(ext):
                    tr.append("{0}{1}/{2}/{3}".format(work_dir, p, s, t))
        trials[p] = tr
        bar.update(1)
    bar.close()
    return trials


def sync():
    # change Trix to your output drive
    out_dir = "{0}/Meta/".format(drive('Trix'))
    if not os.path.exists(out_dir + "data_map.json"):
        trials = initialise(ext='c3d')
        JSONSUtl.write_json(out_dir + "data_map.json", trials)
    else:
        print("Load data map")
        trials = JSONSUtl.load_json(out_dir + "data_map.json")

    # Change this to where you put your IMU folder.
    imu_data = "./IMU/"
    part = [p for p in os.listdir(imu_data) if p.lower().startswith('p')]
    conditions = {p: [c for c in os.listdir("{0}{1}".format(imu_data, p)) if Param.trials_name(c) is not None] for p in part}
    imus = {}
    offset = {}
    for p in conditions:
        # if p != 'P010':
        #     continue
        condition = conditions[p]
        imus[p] = {}
        offset[p] = {}
        for c in condition:
            imus[p][c] = ["{0}{1}/{2}/{3}".format(imu_data, p, c, d) for d in os.listdir("{0}{1}/{2}".format(imu_data, p, c)) if 'forearm' in d.lower() and (d.endswith('_vec3_2_raw.csv') or d.endswith('_vec3_raw.csv'))]
            for im in imus[p][c]:
                spartacus = pd.read_csv(im)
                trial_name = Param.trials_name(c)
                trial_condition = Param.condition(c)
                trial_idx = Param.trial_id(c)
                omc_match = None
                cc = "{0}_{1}_{2}".format(trial_name, trial_condition, trial_idx)
                for mk in trials[p]:
                    sp = os.path.split(mk)
                    spe = sp[1][:sp[1].rindex('.')]
                    trial_name_omc = Param.trials_name(spe)
                    trial_condition_omc = Param.condition(spe)
                    trial_idx_omc = Param.trial_id(spe)
                    try:
                        print("{0} {1} {2} {3}".format(p , trial_name , trial_condition , trial_idx))
                        print("{0} {1} {2} {3}".format(p , trial_name_omc , trial_condition_omc , trial_idx_omc))
                    except TypeError:
                        pass

                    if trial_name == trial_name_omc:
                        if trial_condition == trial_condition_omc:
                            if trial_idx == trial_idx_omc:
                                omc_match = mk
                                print(mk)
                                print(im)
                                break

                    pass
                if omc_match is None:
                    offset[p][cc] = np.nan
                    continue

                try:
                    trc = TRC.create_from_c3d(omc_match)
                except ValueError:
                    offset[p][cc] = np.nan
                    continue
                if 'left' in im.lower():
                    print('left')
                    try:
                        lu = trc.marker_set['L_Ulna']
                    except KeyError:
                        try:
                            lu = trc.marker_set['L_Radius']
                        except KeyError:
                            try:
                                lu = trc.marker_set['L_Lat_HumEpicondyle']
                            except KeyError:
                                try:
                                    lu = trc.marker_set['L_Med_HumEpicondyle']
                                except KeyError:
                                    continue

                else:
                    print('right')
                    try:
                        lu = trc.marker_set['R_Ulna']
                    except KeyError:
                        try:
                            lu = trc.marker_set['R_Radius']
                        except KeyError:
                            try:
                                lu = trc.marker_set['R_Lat_HumEpicondyle']
                            except KeyError:
                                try:
                                    lu = trc.marker_set['R_Med_HumEpicondyle']
                                except KeyError:
                                    continue

                cols = [c for c in lu.columns]
                cols.insert(0, 'time')
                lux = np.zeros([len(lu.index), len(cols)])
                lux[:, 0] = trc.x
                lux[:, 1:] = lu.to_numpy()
                luny = Yatsdo(pd.DataFrame(data=lux, columns=cols))
                lun_tdp = luny.get_samples(trc.x)/1000
                lun = np.array([np.linalg.norm(lun_tdp[i, 1:]) for i in range(0, lun_tdp.shape[0])])

                # ru = trc.marker_set['R_Ulna'].to_numpy()/1000
                # plu = PCAModel.pca_rotation(lu)
                # plud = plu.transformed_data
                # pru = PCAModel.pca_rotation(lu)
                # prud = pru.transformed_data
                spartacus['time'] = spartacus['time'] - spartacus['time'].iloc[0]
                try:
                    spy = Yatsdo(spartacus)
                except ValueError:
                    try:
                        spy = Yatsdo(spartacus, fill_data=True)
                    except ValueError:
                        offset[p][cc] = np.nan
                        continue
                d = int(np.floor(spartacus['time'].iloc[-1]/0.01))
                td = [k*0.01 for k in range(0, d)]
                spartacus_tdp = spy.get_samples(td, as_pandas= True)
                ss = spartacus_tdp[["m_acceleration_X", "m_acceleration_Y", "m_acceleration_Z"]].to_numpy()
                # pss = PCAModel.pca_rotation(ss)
                # pssd = pss.transformed_data
                ssn = np.array([np.linalg.norm(ss[i, :]) for i in range(0, ss.shape[0])])

                ipus = interpolate.InterpolatedUnivariateSpline(trc.x, lun)
                ipusd = ipus.derivative()
                ipusdd = ipusd.derivative()
                acc = ipusdd(trc.x)
                try:
                    ssnb = Butterworth.butter_low_filter(ssn[100:1000], 2, 100) - 9.81
                except ValueError:
                    offset[p][cc] = np.nan
                    continue
                lunb = Butterworth.butter_low_filter(acc[100:1000], 2, 100)
                cross_corr_scipy = signal.correlate(lunb, ssnb, mode='full')
                lags = signal.correlation_lags(lunb.size, ssnb.size, mode='full')


                aligned_lag_index = np.argmax(cross_corr_scipy)
                aligned_lag = lags[aligned_lag_index]
                print(aligned_lag)
                offset[p][cc] = aligned_lag
                plt.figure()
                plt.title(p+" "+trial_name+" "+trial_condition+" "+trial_idx)
                plt.plot(ssnb, color='blue')
                plt.plot(lunb, color='red')
                plt.show()

                pass
    df = pd.DataFrame(data=offset)
    df.to_csv(out_dir+"Sync.csv")


if __name__ == '__main__':
    sync()

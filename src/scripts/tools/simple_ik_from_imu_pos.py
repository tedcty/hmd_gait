import numpy as np
import pandas as pd
from ptb.util.gait.helpers import OsimHelper
from ptb.util.io.mocap.file_formats import TRC
from ptb.util.data import StorageIO

import copy
import os
from opensim import InverseKinematicsTool


class Param:
    @staticmethod
    def trials_name(tx):
        trials_list = {"Combination": ["Combination", "Combo", "Comb"],
                       "Define": ["Define"],
                       "Free": ["Free"],
                       "Straight": ["Straight", "Str"],
                       "Obstacle": ["Obstacle", "Osb", 'obst'],
                       "Reactive": ["Reactive", 'Rea'],
                       "Stairs": ["Stairs", "Stair"],
                       "Static": ['Stat', "Cal"],
                       "Test": ['test']
                       }

        for s in trials_list:
            for c in trials_list[s]:
                if c.lower() in tx.lower():
                    return s
        return None

    @staticmethod
    def condition(tx):
        con_list = {
            "Normal": ["Norm"],
            "AR": ["AR"],
            "VR": ["VR"]
        }

        for s in con_list:
            for c in con_list[s]:
                if c.lower() in tx.lower():
                    return s
        return "Normal"

    @staticmethod
    def trial_id(tx):
        c = tx.strip()[-1]
        try:
            a = int(c)
            return "T{0:01d}".format(a)
        except ValueError:
            return "NA"

    xsens = {
        "Marker": ["LPSH", "RPSH", "LUA3", "RUA3", "LFAsuperior", "RFAsuperior", "Sacrum", "CLAV", "LTH1", "RTH1",
                   "LTB3", "RTB3", "LMT5", "RMT5"],
        "IMU": ["LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm", "LeftForeArm", "RightForeArm",
                "Pelvis", "T8", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot",
                "RightFoot"]
    }


def create_trc_file():
    test = "M:/Mocap/Movella_Re/P025/Straight Normal 01/LeftForeArm_imu_vec3_2.csv"
    est = "M:/Mocap/Movella_Re/P025/Straight Normal 01/"
    e = [f for f in os.listdir(est) if f.split('_')[0] in Param.xsens['IMU']]
    tilename = "M:/Mocap/Movella_Re/P025/Straight Normal 01_imu.trc"
    imu = os.path.split(test)[1].split("_")[0]
    imu_idx = Param.xsens['IMU'].index(imu)
    marker = Param.xsens['Marker'][imu_idx]
    df = pd.read_csv(test)
    pos = ['m_position_X', 'm_position_Y', 'm_position_Z']
    timex = ['time']
    idx = 1
    pos_df = df[pos]
    time_series = df[timex]
    time_series = time_series - time_series.iloc[0, 0]
    npdf = np.zeros([time_series.shape[0], 5])
    npdf[:, 1] = time_series.to_numpy()[:, 0]
    npdf[:, 2:] = pos_df.to_numpy()
    npdf[:, 0] = [i + 1 for i in range(0, time_series.shape[0])]
    cols = ['Frame', 'Time', '{0}_X{1}'.format(marker, idx), '{0}_Y{1}'.format(marker, idx),
            '{0}_Z{1}'.format(marker, idx)]
    df_markers = pd.DataFrame(data=npdf, columns=cols)
    trc = TRC.create_from_panda_dataframe(df_markers, tilename)


if __name__ == '__main__':
    skips = {}
    # skips = {"P025_Combination_Normal_T1", "P025_Free_VR_T2", "P025_Straight_AR_T2", "P025_Reactive_AR_T1"}
    particpant = 'P018'
    out_dir = 'I:/Meta/metaik/{0}/'.format(particpant)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pfile = "M:/Mocap/{0}/".format(particpant)
    osm = "M:/Models/{0}.osim".format(particpant)
    osm_model = OsimHelper(osm)
    osm_model_markers = copy.deepcopy(osm_model.markerset)
    markers = [o for o in osm_model_markers]

    ik_template = "M:/Models/IK_Setup.xml"
    ik = InverseKinematicsTool(ik_template)
    ikset = ik.get_IKTaskSet()
    ik_markers = {i.getName(): i for i in ikset}
    ik.setModel(osm_model.osim_model.osim)
    session = [f for f in os.listdir(pfile) if 'session' in f.lower()]
    c3d_files = [f for f in os.listdir("{0}{1}".format(pfile, session[0])) if f.endswith('.c3d')]
    for f in c3d_files:
        cf = f[:f.rindex('.c3d')]
        cname = Param.trials_name(cf)
        ccon = Param.condition(cf)
        cidx = Param.trial_id(cf)
        trial_folder = "{0}/{1}_{2}_{3}_{4}/".format(out_dir, particpant, cname, ccon, cidx)
        trial_name = "{1}_{2}_{3}_{4}".format(out_dir, particpant, cname, ccon, cidx)
        if trial_name in skips:
            continue
        print(trial_folder)
        if not os.path.exists(trial_folder):
            os.makedirs(trial_folder)
        trc = TRC.create_from_c3d("{0}{1}/{2}".format(pfile, session[0], f))
        print("Reorient markers")
        trc.z_up_to_y_up()
        n0 = [m for m in trc.marker_set if particpant in m]
        if len(n0) == 0:
            n0 = [m for m in trc.marker_set]
        try:
            n = [m for m in n0 if m.split(':')[1] in markers]
        except IndexError:
            n = [m for m in n0 if m in markers]
        print("Find NaNs in markers")
        try:
            nans = {m.split(':')[1]: np.isnan(trc.marker_set[m]) for m in n}
        except IndexError:
            nans = {m: np.isnan(trc.marker_set[m]) for m in n}

        print("Remove Subject Label")
        try:
            trc.marker_set = {m.split(':')[1]: trc.marker_set[m] for m in n}
        except IndexError:
            trc.marker_set = {m: trc.marker_set[m] for m in n}

        mpx = [m for m in trc.marker_set]
        print("Check for last empty column")
        if np.sum(np.isnan(trc.marker_set[mpx[-1]].iloc[0, :])) > 0:
            m0 = mpx[-1]

            m1 = None
            for i in range(0, len(mpx) - 1):
                if np.sum(np.isnan(trc.marker_set[mpx[i]].iloc[0, :])) == 0:
                    m1 = mpx[i]
                    break
            if m1 is not None:
                m0pop = trc.marker_set.pop(mpx[-1])
                m1pop = trc.marker_set.pop(m1)
                trc.marker_set[m0] = m0pop
                trc.marker_set[m1] = m1pop
        mpx = [m for m in trc.marker_set]
        if np.sum(np.isnan(trc.marker_set[mpx[-1]].to_numpy())) > 0:
            zeros = np.zeros(trc.marker_set[mpx[-1]].shape)
            trc.marker_set["zero"] = pd.DataFrame(data=zeros, columns=["X100", "Y100", "Z100"])

        trc.update_from_markerset()
        task = "{0}/{1}_{2}_{3}_{4}.trc".format(trial_folder, particpant, cname, ccon, cidx)
        print("Write new trc file")
        trc.write(task)
        ik.set_marker_file(task)

        print("Split data")
        check = {m: np.array([np.sum(nans[m].iloc[i, :]) for i in range(0, nans[m].shape[0])]) for m in nans}
        df = pd.DataFrame(data=check)
        frames = [np.sum(df.iloc[d, :]) for d in range(0, df.shape[0])]
        splits = []
        st = -1
        ed = -1
        for frame in range(0, len(frames)):
            if st == -1 and frames[frame] == 0:
                st = frame
            elif st != -1 and frames[frame] != 0:
                splits.append([st, frame])
                st = -1
            elif st != -1 and frames[frame] == 0:
                ed = frame
        if st != -1 and ed != -1 and st < ed:
            splits.append([st, ed])
        print("Run IK")
        if len(splits) == 0:
            pass
        for s in splits:
            task = "{1}_{2}_{3}_{4}_{5}-{6}".format(trial_folder, particpant, cname, ccon, cidx, s[0], s[1])
            print("Running {0}".format(task))
            ik.set_output_motion_file(
                "{0}{1}_{2}_{3}_{4}_{5}-{6}.sto".format(trial_folder, particpant, cname, ccon, cidx, s[0], s[1]))
            ik.setStartTime(trc.data[s[0], 1])
            ik.setEndTime(trc.data[s[1], 1])
            try:
                ik.run()
            except RuntimeError:
                pass
            pass

        pass
    pass
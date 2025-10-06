import numpy as np
import pandas as pd
from ptb.util.gait.helpers import OsimHelper
from ptb.util.io.mocap.file_formats import TRC
from ptb.util.gait.analysis import IK
from src.util.meta import Param
import copy
import os
from opensim import InverseKinematicsTool, IKMarkerTask

participants = ["P001", "P010", "P011", "P012", "P014",
                "P015", "P016", "P017", "P018", "P019",
                "P020", "P021", "P025", "P026", "P032",
                "P043"]

def enable_disable_markers(ik_tool, marker_name, enable):
    ikset = ik_tool.get_IKTaskSet()
    k = {i.getName(): i for i in ikset}
    k[marker_name].setApply(enable)
    # for i in k:
    #     print(i)
    #     print(k[i].getApply())
    #     print(k[i].getWeight())
    #     k[i].setApply(False)
    #     pass
    # ikset2 = ik.get_IKTaskSet()
    # k = [i for i in ikset2]
    # for i in k:
    #     print(i.getName())
    #     print(i.getApply())
    #     print(i.getWeight())
    #     pass

def process_c3d_to_trc_slices():
    skips = {"P025_Combination_Normal_T1", "P025_Free_VR_T2", "P025_Straight_AR_T2", "P025_Reactive_AR_T1"}
    particpant = 'P025'
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
        trc = TRC.create_from_c3d("{0}{1}/{2}".format(pfile, session[0],f))
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
        # pop_n = [m for m in trc.marker_set if m not in n]# filter names
        # for n in pop_n:
        #     trc.marker_set.pop(n)
        print("Remove Subject Label")
        try:
            trc.marker_set = {m.split(':')[1]: trc.marker_set[m] for m in n}
        except IndexError:
            trc.marker_set = {m: trc.marker_set[m] for m in n}

        mpx = [m for m in  trc.marker_set]
        print("Check for last empty column")
        if np.sum(np.isnan(trc.marker_set[mpx[-1]].iloc[0, :])) >0:
            m0 = mpx[-1]

            m1 = None
            for i in range(0, len(mpx)-1):
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
            ik.set_output_motion_file("{0}{1}_{2}_{3}_{4}_{5}-{6}.sto".format(trial_folder, particpant, cname, ccon, cidx, s[0], s[1]))
            ik.setStartTime(trc.data[s[0], 1])
            ik.setEndTime(trc.data[s[1], 1])
            try:
                ik.run()
            except RuntimeError:
                pass
            pass

        # n0 = np.array([np.sum(n[i, : ]) for i in range(0, n.shape[0])])
        pass
    pass

if __name__ == '__main__':
    # todo
    # get list of participant and trials
    # rerun the IK from c3d.
    process_c3d_to_trc_slices()
    # model_dir = "M:/temp/"
    # mt = [m for m in os.listdir(model_dir) if m.lower().startswith("p")]
    # for m in mt:
    #     particpant = m[:m.rindex(".")]
    #     mf = [s for s in os.listdir("M:/Mocap/{0}/".format(particpant)) if 'session' in s.lower()]
    #     opens_model = OsimHelper("{0}{1}.osim".format(model_dir, particpant))
    #     trials = [t for t in os.listdir("M:/Mocap/{0}/{1}".format(particpant, mf[0])) if t.endswith("c3d")]
    #     for t in trials:
    #         kk = ""
    #         pass
    #         IK.run_from_c3d(
    #             wkdir="M:/Mocap/test/{0}/".format(particpant),
    #             model=opens_model.osim_model.osim,
    #             trial_c3d="M:/Mocap/{0}/{1}/{2}".format(particpant, mf[0], t))
    #     # if particpant == "P048":
    #     #     opens_model = OsimHelper("{0}{1}.osim".format(model_dir, particpant))
    #     #
    #     #     IK.run_from_c3d(
    #     #         wkdir="M:/Mocap/test/",
    #     #         model=opens_model.osim_model.osim,
    #     #         trial_c3d="M:/Mocap/{0}/New Session/P048 Cal 06.c3d".format(particpant))
    pass
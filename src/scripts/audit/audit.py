import os
import pandas as pd
from ptb.core import Yatsdo
import numpy as np
from shutil import copyfile


def trials_name(tx):
    trials_list = {"Combination": ["Combination", "Combo", "Comb"],
                   "Define": ["Define"],
                   "Free": ["Free"],
                   "Straight": ["Straight"],
                   "Obstacle": ["Obstacle"],
                   "Reactive": ["Reactive"],
                   "Stairs": ["Stairs"]
                   }

    for s in trials_list:
        for c in trials_list[s]:
            if c.lower() in tx.lower():
                return s
    return None

def trial_id(tx):
    if tx.strip().endswith("1"):
        return "T01"
    elif tx.strip().endswith("2"):
        return "T02"
    elif tx.strip().endswith("3"):
        return "T03"
    return "NA"


def worker(imu_folder, out_folder, trials, p):
    for t in trials[p]:
        folder = "{0}{1}/{2}".format(imu_folder, p, t)
        out_trial = trials_name(t)
        trial = trial_id(t)
        out = "{0}{1}/{2}/{3}/".format(out_folder, p, out_trial, trial)
        if not os.path.exists(out):
            os.makedirs(out)
        cc = [c for c in os.listdir("{0}{1}/{2}".format(imu_folder, p, t)) if c.startswith('joint_angles')]
        for c in cc:
            try:
                print("{0}{1}/{2}/{3}".format(imu_folder, p, t, c))
                c2 = c
                if "_2_" in c:
                    c2 = c.replace("_2_", "_")
                outfileName = "{0}{1}".format(out, c2)
                print(outfileName)
                if os.path.exists(outfileName):
                    print(outfileName + "...Done!")
                    continue

                df = pd.read_csv("{0}{1}/{2}/{3}".format(imu_folder, p, t, c))
                df.iloc[:, 0] = df.iloc[:, 0] - df.iloc[0, 0]
                y = Yatsdo(df)
                frames = int(np.floor(df.iloc[-1, 0] / (1 / 100)))
                points = [ti * 0.01 for ti in range(0, frames)]
                pout = y.get_samples(points, as_pandas=True)
                pout.to_csv(outfileName, index=False)

                pass
            except Exception:
                continue
        pass

def IMU():
    imu_folder = "M:/Mocap/Movella_Re/"
    out_folder = "M:/Audit/IMU/"
    particip = [f for f in os.listdir(imu_folder)]
    particip.sort()
    trials = {p: [t for t in os.listdir("{0}{1}".format(imu_folder, p)) if 'figure' not in t.lower()] for p in particip}
    for p in trials:
        worker(imu_folder, out_folder, trials, p)
        pass

def reading_enf_file():
    pass

if __name__ == '__main__':
    mocap_folder = "M:/Mocap/"
    participant = ['P{0:03d}'.format(i) for i in range(1, 36)]
    session = {p: [ o for o in os.listdir("{0}{1}".format(mocap_folder, p)) if os.path.isdir("{0}{1}/{2}".format(mocap_folder, p, o))][0] for p in participant}
    recommend_full = {}
    output = 'C:/Users/tyeu008/Uni of Auckland Dropbox/hmd-gait/Shares/Dataset/Pre-Release/OMC/'
    for p in participant:
        print(p)
        trials = [o for o in os.listdir("{0}{1}/{2}".format(mocap_folder, p, session[p])) if o.endswith("Trial.enf")]
        recommend = []
        for f in trials:
            infile = "{0}{1}/{2}/{3}".format(mocap_folder, p, session[p], f)
            fw = open(infile)
            yx = fw.read()
            fw.close()
            try:
                x = [k for k in yx.split('\n') if 'DESCRIPTION' in k]
                des = int(x[0].split("=")[1])
                if des == 1:
                    c3d_file = "{0}.c3d".format(f[:f.rindex('.Trial.enf')])
                    if os.path.exists("{0}{1}/{2}/{3}".format(mocap_folder, p, session[p], c3d_file)):
                        recommend.append(c3d_file)
                pass
            except IndexError:
                continue
            except ValueError:
                continue
        recommend_full[p] = recommend
        pass
    # for r in recommend_full:
    #     rlis = recommend_full[r]
    #     if r == 'P001':
    #         continue
    #     if len(rlis) > 0:
    #         for l in rlis:
    #             print("{0}{1}/{2}/{3}".format(mocap_folder, r, session[r], l))
    #             h = "{0}{1}/{2}/{3}".format(mocap_folder, r, session[r], l)
    #             g = "{0}{1}/{2}".format(output, r, l)
    #             copyfile(h, g)
    #             pass

    for r in participant:
        if r == 'P001' or r == 'P009' or len(recommend_full[r]) == 0:
            c3df = [c for c in os.listdir("{0}{1}/{2}/".format(mocap_folder, r, session[r])) if c.endswith('.c3d')]
            for l in c3df:
                print("{0}{1}/{2}/{3}".format(mocap_folder, r, session[r], l))
                h = "{0}{1}/{2}/{3}".format(mocap_folder, r, session[r], l)
                g = "{0}{1}/{2}".format(output, r, l)
                copyfile(h, g)
                pass
            pass
    pass
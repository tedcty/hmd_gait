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


if __name__ == '__main__':
    imu_folder = "M:/Mocap/"
    out_folder = "M:/Audit/OMC/"
    particip = [f for f in os.listdir(imu_folder) if f.startswith('P')]
    particip.sort()
    session = {p: [f for f in os.listdir("{0}{1}".format(imu_folder, p)) if 'sess' in f.lower()]  for p in particip}
    markered = {}
    for p in session:
        idxN = int(p[1:])
        idx = "P{:03d}".format(idxN)
        print(idx)
        if not os.path.exists("{0}{1}".format(out_folder, p)):
            os.makedirs("{0}{1}".format(out_folder, p))
        fx = {}
        for s in session[p]:
            mypath = "{0}{1}/{2}".format(imu_folder, p, s)
            files = [f for f in os.listdir(mypath) if f.endswith('enf')]

            for f in files:
                infile = "{0}{1}/{2}/{3}".format(imu_folder, p, s, f)
                fw = open(infile)
                yx = fw.read()
                fw.close()
                try:
                    x = [k for k in yx.split('\n') if 'MARKERSLABELED' in k]
                    percent = float(x[0].split("=")[1])
                    fx[f] = percent
                except IndexError:
                    continue
            pass
        markered[p] = fx

        pass
    for m in markered:
        df = pd.Series(data=markered[m])
        # df.to_csv("M:/Audit/OMC/{0}_percentage_tracked.csv".format(m))
        print("{0}: {1}".format(m, np.median(df)))
    print()
    pass
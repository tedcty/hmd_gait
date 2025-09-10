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


def worker():
    for t in trials[p]:
        folder = "{0}{1}/{2}".format(imu_folder, p, t)
        out_trial = trials_name(t)
        trial = trial_id(t)
        out = "{0}{1}/{2}/{3}/".format(out_folder, p, out_trial, trial)
        if not os.path.exists(out):
            os.makedirs(out)
        cc = [c for c in os.listdir("{0}{1}/{2}".format(imu_folder, p, t)) if c.endswith('_raw.csv')]
        for c in cc:
            try:
                print("{0}{1}/{2}/{3}".format(imu_folder, p, t, c))
                c2 = c
                if "_2_" in c:
                    c2 = c.replace("_2_", "_")
                outfileName = "{0}{1}".format(out, c2)

                print(outfileName)
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


if __name__ == '__main__':
    imu_folder = "M:/Mocap/Movella_Re/"
    out_folder = "M:/Audit/IMU/"
    particip = [f for f in os.listdir(imu_folder)]
    particip.sort()
    trials = {p:[t for t in os.listdir("{0}{1}".format(imu_folder, p)) if 'figure' not in t.lower()] for p in particip}
    for p in trials:
        worker()
        pass
    pass
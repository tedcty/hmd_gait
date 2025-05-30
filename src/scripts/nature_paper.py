import os
import numpy as np
from multiprocessing import Pool
from shutil import copyfile


def worker(args):
    sour = args[0]
    targ = args[1]
    print("copying {0} to {1}".format(sour, targ))
    try:
        copyfile(sour, targ)
        print("copied {0} to {1}".format(sour, targ))
    except IOError:
        print("IO Error copying {0} to {1}".format(sour, targ))

if __name__ == '__main__':
    work_dir = "M:/Mocap/Movella_Re/"
    out_dir = "I:/Meta/"
    participants = [f for f in os.listdir(work_dir)]
    participants_size = []
    print("Scanning for data in: {0}".format(work_dir))
    for p in participants:
        trials = [f for f in os.listdir("{0}{1}".format(work_dir, p))]
        trials_size = []
        for t in trials:
            print("{0}: {1}".format(p, t))
            imus = [f for f in os.listdir("{0}{1}/{2}".format(work_dir, p, t)) if f.endswith(".csv.pkl")]
            imus_size = [os.path.getsize("{0}{1}/{2}/{3}".format(work_dir, p, t, f)) for f in imus]
            trials_size.append(np.sum(imus_size))
            pass
        participants_size.append(np.sum(trials_size))
    part_size = np.sum(participants_size)
    part_kb = part_size/1024.0
    part_MB = part_kb/1024.0
    part_GB = part_MB/1024.0

    print("Data to be copied {0:.2f} GB".format(np.round(part_GB, 2)))

    copier = []



    print("Creating jobs")
    for p in participants:
        trials = [f for f in os.listdir("{0}{1}".format(work_dir, p))]
        for t in trials:
            imus = [f for f in os.listdir("{0}{1}/{2}".format(work_dir, p, t)) if f.endswith(".csv.pkl")]
            for i in imus:
                so = "{0}{1}/{2}/{3}".format(work_dir, p, t, i)
                ta = "{0}{1}/{2}/{3}".format(out_dir, p, t, i)

                if not os.path.exists("{0}{1}/{2}/".format(out_dir, p, t)):
                    os.makedirs("{0}{1}/{2}/".format(out_dir, p, t))
                copier.append([so, ta])
            pass

    print("Copying ...")
    poo = Pool(12)
    ret = poo.map(worker, copier)
    print("Done!")
    pass
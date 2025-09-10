import os
import numpy as np
import pandas as pd
import gzip
import pickle

import ptb
from ptb.core import Yatsdo
from ptb.util.io.helper import StorageIO, StorageType, TRC, JSONSUtl
from ptb.ml.ml_util import MLOperations, MLKeys

def not_in_skip(x):
    skips = ['1 Figures']
    for s in skips:
        if x in s:
            return False
    return True

def is_lower_IMU(x):
    lower = ['Leg', 'Foot', 'Pelvis']
    for l in lower:
        if l.lower() in x.lower():
            return True
    return False

def gen_windows(y: Yatsdo, size=60):
    # 1 sec
    num_winds = [[s, s+size] for s in range(y.shape[0] - size)]
    windows = []
    idx = 1
    for s in num_winds:
        d = y.get_samples(y.data[s[0]:s[1], 0])
        n = np.zeros([d.shape[0], y.data.shape[1]+1])
        n[:, 0] = idx
        n[:, 1:] = d
        windows.append(n)
        idx += 1
    return windows

def export_features(retx, file_pathx):
    with gzip.open(file_pathx, 'wb', compresslevel=9) as fx:
        pickle.dump(retx, fx)
    pass

def load_features(file_pathx):
    with gzip.open(file_pathx, "rb") as fx:
        loaded_data = pickle.load(fx)
    return loaded_data

if __name__ == '__main__':
    root = "M:/Mocap/Movella_Re/"
    pid = os.listdir(root)
    pid.sort()
    kid = {p: [q for q in os.listdir("{0}{1}".format(root, p)) if not_in_skip(q)] for p in pid}
    # total_file_size_mb = []
    # total_file_size_gb = []
    # for p in kid:
    #     for k in kid[p]:
    #         b = [o for o in os.listdir("{0}{1}/{2}".format(root, p, k)) if o.endswith('_vec3_2.csv') if is_lower_IMU(o)]
    #         for l in b:
    #             print()
    #             file_path = "{0}{1}/{2}/{3}".format(root, p, k, l)
    #             print(file_path)
    #             file_size_bytes = os.path.getsize(file_path)
    #             print(f"File size: {file_size_bytes} bytes")
    #
    #             # Optionally, convert to other units
    #             file_size_kb = file_size_bytes / 1024
    #             file_size_mb = file_size_kb / 1024
    #             file_size_gb = file_size_mb / 1024
    #             total_file_size_mb.append(file_size_mb)
    #             total_file_size_gb.append(file_size_gb)
    #             print(f"File size: {file_size_kb:.2f} KB")
    #             print(f"File size: {file_size_mb:.2f} MB")
    #             print(f"File size: {file_size_gb:.2f} GB")
    #         pass
    #     pass
    # total_size_mb = np.sum(total_file_size_mb)
    # total_size_gb = np.sum(total_file_size_gb)
    count = 0
    for p in kid:
        if count > 10:
            break
        for k in kid[p]:
            b = [o for o in os.listdir("{0}{1}/{2}".format(root, p, k)) if o.endswith('_vec3_2_raw.csv') if
                 is_lower_IMU(o)]
            if len(b) == 0:
                b = [o for o in os.listdir("{0}{1}/{2}".format(root, p, k)) if o.endswith('_vec3_raw.csv') if
                     is_lower_IMU(o)]
            for l in b:
                print(p + "> " + k + ": " + l)
                fc = MLKeys.CFCParameters
                table_id_name = fc.name
                filename = "{3}_{0}_{1}_features_{2}".format(l[:-4], k, table_id_name, p)
                file_path_out = "I:/Meta/TestMLFeatures/" + filename + ".pkl.gz"
                if os.path.exists(file_path_out):
                    continue
                file_path = "{0}{1}/{2}/{3}".format(root, p, k, l)
                try:
                    pk = pd.read_csv(file_path)
                except pd.errors.EmptyDataError:
                    try:
                        pk = pd.read_pickle(file_path)
                    except pd.errors.EmptyDataError:
                        continue
                try:
                    y = Yatsdo(pk)
                except ValueError:
                    continue
                w = gen_windows(y)
                ws = np.vstack(w)
                wscols = ['id']
                for yc in y.column_labels:
                    wscols.append(yc)
                psdf = pd.DataFrame(data=ws, columns=wscols)
                # fc = MLKeys.MFCParameters

                efx, param = MLOperations.extract_features_from_x(psdf, fc_parameters=fc, n_jobs=12)
                ret = {'efx': efx, 'param': param}
                file_path = "I:/Meta/TestMLFeatures/" + filename + ".pkl.gz"
                # file_path = "D:/Ted/Meta/TestMLFeatures/" + filename + ".pkl.gz"
                export_features(ret, file_path)
                print("Done.")
                # MLOperations.export_features(efx, param, output_folder="I:/Meta/TestMLFeatures/", filename=filename, table_id=table_id_name)
            print(k)
        print(p)
        count += 1
    pass
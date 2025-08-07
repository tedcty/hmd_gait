from ptb.ml.ml_util import MLOperations, MLKeys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from enum import Enum

class Category(Enum):
    static = 0
    walking = 1
    turning = 2
    cones = 3
    step_over = 4
    ascent = 5
    descent = 6

    def to_string(self):
        return str(self.name)

    @staticmethod
    def get(x:str):
        for i in Category:
            if x.lower() in str(i.name):
                return i
        return None



if __name__ == '__main__':
    k = Category.static

    # X, param = MLOperations.extract_features_from_x(in_data[in_d], n_jobs=2)
    # MLOperations.export_features(efx, param, output_dir, out_file_name, as_csv=True)
    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    # clf.fit(X, y)
    pass
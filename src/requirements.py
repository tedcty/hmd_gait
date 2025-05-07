import os

if __name__ == '__main__':
    print("Getting Requirement:")
    os.system('conda install -c opensim-org opensim')
    os.system('python -m pip install build pandas scipy numpy scikit-learn vtk PySide6 tsfresh')
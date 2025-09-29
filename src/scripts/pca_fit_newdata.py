from gias3.learning.PCA import PCA
from gias3.learning.PCA import loadPrincipalComponents

if __name__ == "__main__":
    # Load trained PCA model
    event = None
    condition = None
    pc = loadPrincipalComponents(f"IMU_{event}_{condition}_top100_pca.pc")

    # Load feature data for new participants

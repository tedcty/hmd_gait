from enum import Enum
import copy


class MetaMarkerSet(Enum):
    torso = ["L_Acromion", "R_Acromion", "Sternum"]
    pelvis = ["R_ASIS", "L_ASIS", "R_PSIS", "L_PSIS"]
    left_upper_arm = ["L_Lat_HumEpicondyle", "L_Med_HumEpicondyle"]
    right_upper_arm = ["R_Lat_HumEpicondyle", "R_Med_HumEpicondyle"]

    left_upper_leg = ["L_MedKnee", "L_LatKnee"]
    right_upper_leg = ["R_MedKnee", "R_LatKnee"]

    left_lower_leg = ["L_FibHead", "L_MidShank", "L_MedAnkle", "L_LatAnkle"]
    right_lower_leg = ["R_FibHead", "R_MidShank", "R_MedAnkle", "R_LatAnkle"]

    @staticmethod
    def get(m):
        return copy.deepcopy(m.value)
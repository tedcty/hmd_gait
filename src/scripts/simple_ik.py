from ptb.util.gait.helpers import OsimHelper
from ptb.util.gait.analysis import IK
import os

if __name__ == '__main__':
    # todo
    # get list of participant and trials
    # rerun the IK from c3d.
    model_dir = "M:/temp/"
    mt = [m for m in os.listdir(model_dir) if m.lower().startswith("p")]
    for m in mt:
        particpant = m[:m.rindex(".")]
        if particpant == "P048":
            opens_model = OsimHelper("{0}{1}.osim".format(model_dir, particpant))
            IK.run_from_c3d(
                wkdir="M:/Mocap/test/",
                model=opens_model.osim_model.osim,
                trial_c3d="M:/Mocap/{0}/New Session/P048 Cal 06.c3d".format(particpant))
    pass
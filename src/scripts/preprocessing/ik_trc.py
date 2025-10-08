from opensim import InverseKinematicsTool
from ptb.util.data import TRC
import os

if __name__ == "__main__":
    
    wkdir = "Z:/Upper Body/Mocap"  # the working dir where your trc files are

    template = "Z:/Models/IK_Setup.xml"  # NOTE: CHECK IF THIS IS THE CORRECT TEMPLATE
    ik = InverseKinematicsTool(template)  # loading the tool

    osim_dir = "Z:/Models"  # the directory containing your .osim model files

    out_dir = "Z:/Upper Body/Kinematics"  # the output root folder containing the kinematics files to process

    # Search for trc files in each of the subfolders in the working directory
    for root, dirs, files in os.walk(wkdir):
        for filename in files:
            if filename.endswith(".trc"):
                trc_file = os.path.join(root, filename)
                # Find model for this subject
                p_id = os.path.basename(root)  # assuming folder name is participant ID
                model = os.path.join(osim_dir, f"{p_id}.osim")
                # Set marker file
                ik.set_marker_file(trc_file)
                b = ik.get_output_motion_file()
                bf = os.path.join(out_dir, p_id, filename.replace(".trc", ".mot"))  # Same filename (just replacing .trc with .mot) but in the out_dir folder
                ik.set_output_motion_file(bf)  # set mot filename for saving the results

                trc = TRC.read(trc_file)
                x = trc[:, 1]  # get time column
                ik.setStartTime(x[0])  # set start time
                ik.setEndTime(x[-1])  # set end time

                try:
                    ik.run()
                except RuntimeError:
                    pass
                pass
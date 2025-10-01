from ptb.util.data import TRC
import os

if __name__ == "__main__":

    wkdir = "Z:/Upper Body/Mocap/P003"                        # your working dir where your trc files are

    trial_c3d = "Z:/Upper Body/Mocap/P003"                    # the root folder containing the c3d files you want to process

    # Search for c3d files in the root folder
    for root, dirs, files in os.walk(trial_c3d):
        for file in files:
            if file.endswith(".c3d"):
                c3d_path = os.path.join(root, file)
                trc_path = c3d_path.replace(".c3d", ".trc")
                print(f"Processing {c3d_path} to {trc_path}")
                
                # Generate TRC file from C3D
                trc = TRC.create_from_c3d(c3d_path)
                trc.z_up_to_y_up()
                
                # Ensure output directory exists
                output_dir = os.path.dirname(trc_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Write TRC file
                trc.write(trc_path)
                print(f"Saved TRC to {trc_path}")
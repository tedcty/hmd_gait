from opensim import InverseKinematicsTool
from ptb.util.data import TRC
import os
from datetime import datetime

if __name__ == "__main__":
    
    wkdir = "Z:/Upper Body/Mocap"  # the working dir where your trc files are

    template = "Z:/Models/IK_Setup.xml"  # XML settings
    ik = InverseKinematicsTool(template)  # loading the tool

    osim_dir = "Z:/Models"  # the directory containing your .osim model files

    out_dir = "Z:/Upper Body/Kinematics"  # the output root folder containing the kinematics files to process

    excluded_participants = ["P001" ,"P010", "P011", "P012", "P014", "P016", "P017", "P019", "P026", "P043"]  # list of participants to exclude

    # Create log file
    log_file = os.path.join(out_dir, f"ik_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(log_file, 'w') as log:
        log.write(f"IK Processing Log - Started: {datetime.now()}\n")
        log.write("="*50 + "\n\n")
        
        # Search for trc files in each of the subfolders in the working directory
        for root, dirs, files in os.walk(wkdir):
            p_id = os.path.basename(root)  # assuming folder name is participant ID
            
            # Skip excluded participants
            if p_id in excluded_participants:
                continue
                
            # Group TRC files by their base name (without _Reconstructed)
            trc_groups = {}
            for filename in files:
                if filename.endswith(".trc"):
                    if filename.endswith("_Reconstructed.trc"):
                        base_name = filename.replace("_Reconstructed.trc", ".trc")
                        if base_name not in trc_groups:
                            trc_groups[base_name] = {}
                        trc_groups[base_name]['reconstructed'] = filename
                    else:
                        if filename not in trc_groups:
                            trc_groups[filename] = {}
                        trc_groups[filename]['original'] = filename
            
            # Process each group - prefer reconstructed if available
            for base_filename, file_info in trc_groups.items():
                # Choose reconstructed if available, otherwise original
                if 'reconstructed' in file_info:
                    actual_filename = file_info['reconstructed']
                    file_type = "reconstructed"
                elif 'original' in file_info:
                    actual_filename = file_info['original']
                    file_type = "original"
                else:
                    continue  # Skip if neither found
                    
                trc_file = os.path.join(root, actual_filename)
                model = os.path.join(osim_dir, f"{p_id}.osim")
                
                # Set marker file
                ik.set_marker_file(trc_file)
                b = ik.get_output_motion_file()
                
                # Output filename uses the base name (without _Reconstructed)
                bf = os.path.join(out_dir, p_id, base_filename.replace(".trc", ".mot"))
                ik.set_output_motion_file(bf)

                trc = TRC.read(trc_file)
                x = trc.data[:, 1]  # get time column
                ik.setStartTime(x[0])  # set start time
                ik.setEndTime(x[-1])  # set end time

                try:
                    ik.run()
                    success_msg = f"SUCCESS: {p_id} - {base_filename} (used {file_type})"
                    print(success_msg)
                    log.write(f"{datetime.now().strftime('%H:%M:%S')} - {success_msg}\n")
                except RuntimeError as e:
                    error_msg = f"ERROR: {p_id} - {base_filename} (used {file_type}): {e}"
                    print(error_msg)
                    log.write(f"{datetime.now().strftime('%H:%M:%S')} - {error_msg}\n")
                    
        log.write(f"\nProcessing completed: {datetime.now()}\n")
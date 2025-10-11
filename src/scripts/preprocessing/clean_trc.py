from ptb.util.data import TRC
import os
import glob


def clean_column_names(col_labels):
    """
    Clean column names by removing the prefix before the colon and the suffix after the underscore.
    For example, 'P003a:Head_X13' becomes 'Head'
    Also handles duplicate marker names by making only the first occurrence visible.
    """
    cleaned_names = []
    seen_markers = {}  # Track how many times we've seen each marker
    
    for col in col_labels:
        if col in ['Frame#', 'Time']:
            # Keep Frame# and Time as is
            cleaned_names.append(col)
        else:
            # Process all other columns (with or without colon)
            if ':' in col:
                # Remove everything before and including the colon
                after_colon = col.split(':', 1)[1]
            else:
                # No colon, use the whole column name
                after_colon = col
            
            # Remove the suffix (everything from the last underscore onwards)
            # Find the last underscore followed by X, Y, Z, or numbers
            parts = after_colon.split('_')
            if len(parts) > 1:
                # Check if the last part looks like a coordinate suffix (X13, Y13, Z13, X1, Y1, Z1, etc.)
                last_part = parts[-1]
                if (last_part.startswith(('X', 'Y', 'Z')) or 
                    last_part.isdigit() or 
                    (len(last_part) > 1 and last_part[0] in 'XYZ' and last_part[1:].isdigit())):
                    # Remove the last part (the coordinate suffix)
                    marker_name = '_'.join(parts[:-1])
                else:
                    # Keep the whole thing if it doesn't look like a coordinate suffix
                    marker_name = after_colon
            else:
                marker_name = after_colon
            
            # Handle duplicate marker names (X, Y, Z coordinates)
            if marker_name in seen_markers:
                # We've seen this marker before, make it blank
                seen_markers[marker_name] += 1
                cleaned_names.append('')
            else:
                # First time seeing this marker, keep the name
                seen_markers[marker_name] = 1
                cleaned_names.append(marker_name)
    
    return cleaned_names


def needs_processing(trc, keep_strings):
    """
    Check if the TRC file needs processing based on:
    1. Whether there are columns to filter out
    2. Whether there are column names that need cleaning (have colons or coordinate suffixes)
    """
    # Check if filtering is needed (are there columns that would be removed?)
    keep_cols = [c for c in trc.col_labels if c in ['Frame#', 'Time'] or any(s in c for s in keep_strings)]
    filtering_needed = len(keep_cols) != len(trc.col_labels)
    
    # Check if column name cleaning is needed
    cleaning_needed = False
    for col in keep_cols:
        if col not in ['Frame#', 'Time']:
            # Check if this column has a colon prefix or coordinate suffix that needs cleaning
            if ':' in col:
                cleaning_needed = True
                break
            # Check for coordinate suffixes
            parts = col.split('_')
            if len(parts) > 1:
                last_part = parts[-1]
                if (last_part.startswith(('X', 'Y', 'Z')) or 
                    last_part.isdigit() or 
                    (len(last_part) > 1 and last_part[0] in 'XYZ' and last_part[1:].isdigit())):
                    cleaning_needed = True
                    break
    
    return filtering_needed or cleaning_needed


def filter_trc_columns(trc, keep_strings):
    """
    Keep only the TRC columns that contain any of the specified substrings (a.k.a. the 36 markers, Time, and Frame#).
    """
    # Keep 'Frame#' and 'Time' columns
    keep_cols = [c for c in trc.col_labels if c in ['Frame#', 'Time'] or any(s in c for s in keep_strings)]
    
    # Get the indices of the columns to keep
    keep_indices = [trc.col_labels.index(col) for col in keep_cols]

    # Filter the TRC data using column indices
    filtered_data = trc.data[:, keep_indices]

    # Clean the column names
    cleaned_col_names = clean_column_names(keep_cols)
    
    return filtered_data, cleaned_col_names


def write_trc_file(filename, data, col_labels, original_trc):
    """
    Manually write TRC file with proper format
    """
    # Calculate number of markers (excluding Frame# and Time)
    # Count non-empty, non-Frame#, non-Time labels
    marker_count = sum(1 for label in col_labels if label not in ['Frame#', 'Time', ''])
    num_markers = marker_count  # Each unique marker name represents one marker
    
    with open(filename, 'w') as f:
        # Write header line 1
        f.write(f"PathFileType\t3\t(X/Y/Z)\t{filename}\n")
        
        # Write header line 2 - get values from original TRC if available
        data_rate = getattr(original_trc, 'sample_rate', 100)
        camera_rate = getattr(original_trc, 'camera_rate', 100)
        num_frames = data.shape[0]
        units = getattr(original_trc, 'units', 'mm')
        
        f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{data_rate}\t{camera_rate}\t{num_frames}\t{num_markers}\t{units}\t{data_rate}\t1\t{num_frames}\n")
        
        # Write column headers (marker names)
        header_line = "\t".join(col_labels)
        f.write(header_line + "\n")
        
        # Write coordinate labels (X1, Y1, Z1, X2, Y2, Z2, etc.)
        coord_labels = []
        
        for i, col in enumerate(col_labels):
            if col == 'Frame#':
                coord_labels.append('')
            elif col == 'Time':
                coord_labels.append('')
            else:
                # For all marker columns, assign sequential X, Y, Z coordinates
                coord_index = i - 2  # Subtract Frame# and Time
                coord_type = ['X', 'Y', 'Z'][coord_index % 3]
                marker_num = (coord_index // 3) + 1
                coord_labels.append(f"{coord_type}{marker_num}")
        
        coord_line = "\t".join(coord_labels)
        f.write(coord_line + "\n")
        
        # Write data with preserved formatting
        for row in data:
            formatted_values = []
            for i, val in enumerate(row):
                if col_labels[i] == 'Frame#':
                    # Frame# should be integer
                    formatted_values.append(str(int(val)))
                elif col_labels[i] == 'Time':
                    # Preserve original Time formatting by removing trailing zeros
                    if isinstance(val, (int, float)):
                        # Format with enough precision, then remove trailing zeros
                        time_str = f"{val:.10f}".rstrip('0').rstrip('.')
                        if '.' not in time_str:
                            time_str += '.0'  # Ensure at least one decimal for consistency
                        formatted_values.append(time_str)
                    else:
                        formatted_values.append(str(val))
                else:
                    # For marker coordinates, preserve original precision
                    if isinstance(val, (int, float)):
                        # Convert to string and preserve original decimal places
                        # Format with high precision first, then clean up
                        val_str = f"{val:.10f}".rstrip('0').rstrip('.')
                        # If it became an integer, add one decimal place for consistency
                        if '.' not in val_str:
                            val_str += '.0'
                        formatted_values.append(val_str)
                    else:
                        formatted_values.append(str(val))
            
            row_str = "\t".join(formatted_values)
            f.write(row_str + "\n")


def process_trc_file(trc_file_path, keep_strings):
    """
    Process a single TRC file
    """
    try:
        # Read TRC file
        trc = TRC.read(trc_file_path)
        
        # Check if processing is needed
        if not needs_processing(trc, keep_strings):
            print(f"  {os.path.basename(trc_file_path)} - already clean, skipping")
            return True
        else:
            print(f"  {os.path.basename(trc_file_path)} - processing...")
            
            # Filter TRC columns
            filtered_data, cleaned_col_names = filter_trc_columns(trc, keep_strings)
            
            # Save the filtered TRC file (overwrite original)
            write_trc_file(trc_file_path, filtered_data, cleaned_col_names, trc)
            print(f"    Completed: {len(trc.col_labels)} -> {len(cleaned_col_names)} columns")
            return True
            
    except Exception as e:
        print(f"  ERROR processing {os.path.basename(trc_file_path)}: {e}")
        return False


if __name__ == "__main__":

    # Configuration
    wkdir = "Z:/Upper Body/Mocap"
    
    # List of participants to exclude (add participant IDs here)
    exclude_participants = ["P001" ,"P010", "P011", "P012", "P014", "P016", "P017", "P019", "P026", "P043"]  # Add participant IDs to exclude
    
    # Markers to keep
    keep_strings = ['Head', 'Sternum', 'R_Acromion', 'L_Acromion', 'R_ASIS', 'L_ASIS', 'R_PSIS', 'L_PSIS', 
                    'L_Lat_HumEpicondyle', 'L_Med_HumEpicondyle', 'L_Radius', 'L_Ulna', 'R_Lat_HumEpicondyle', 
                    'R_Med_HumEpicondyle', 'R_Radius', 'R_Ulna', 'L_LatKnee', 'L_MedKnee', 'L_FibHead', 'L_MidShank',
                    'R_LatKnee', 'R_MedKnee', 'R_FibHead', 'R_MidShank', 'L_DP1', 'L_MT2', 'L_MT5', 'L_LatAnkle', 
                    'L_MedAnkle', 'L_Heel', 'R_DP1', 'R_MT2', 'R_MT5', 'R_LatAnkle', 'R_MedAnkle', 'R_Heel']
    
    # Find all participant directories
    participant_dirs = [d for d in os.listdir(wkdir) 
                       if os.path.isdir(os.path.join(wkdir, d)) 
                       and d.startswith('P') 
                       and d not in exclude_participants]
    
    print(f"Found {len(participant_dirs)} participant directories to process")
    print(f"Excluding: {exclude_participants}")
    print(f"Processing: {sorted(participant_dirs)}\n")
    
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    # Process each participant directory
    for participant_id in sorted(participant_dirs):
        participant_path = os.path.join(wkdir, participant_id)
        print(f"Processing participant: {participant_id}")
        
        # Find all .trc files in this participant's directory (including subdirectories)
        trc_files = glob.glob(os.path.join(participant_path, "**/*.trc"), recursive=True)
        
        if not trc_files:
            print(f"  No .trc files found in {participant_id}")
            continue
        
        print(f"  Found {len(trc_files)} .trc files")
        
        # Process each TRC file
        for trc_file in trc_files:
            total_files += 1
            if process_trc_file(trc_file, keep_strings):
                processed_files += 1
            else:
                failed_files += 1
        
        print()  # Empty line between participants
    
    # Summary
    print("="*50)
    print(f"SUMMARY:")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {failed_files}")
    print(f"Excluded participants: {exclude_participants}")

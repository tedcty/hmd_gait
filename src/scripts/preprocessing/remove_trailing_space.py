# Removes trailing whitespace and tabs from a TRC file, while preserving the first 5 lines.

participant_id = "P013"  # Replace with the actual participant ID
session_id = "Obstacles VR 1_Reconstructed"  # Replace with the actual session ID

input_path = fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc"
output_path = fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc"

with open(input_path, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# Keep the first 5 lines as-is (TRC header)
cleaned_lines = lines[:5]

# Clean lines from line 6 onwards
for line in lines[5:]:
    # Strip trailing whitespace and tabs, then add a single newline
    cleaned_line = line.rstrip('\t\r\n') + '\n'
    cleaned_lines.append(cleaned_line)

# Write to new file
with open(output_path, 'w', encoding='utf-8') as outfile:
    outfile.writelines(cleaned_lines)

print(f"Cleaned file written to: {output_path}")

import re

def convert_srt_to_dialogue(input_data):
    # Initialize variables to store the final dialogue
    dialogue = ""

    # Split the input data by lines
    srt_content = input_data.splitlines()

    # Regular expression to match timestamp lines (e.g., 00:00:00,000 --> 00:00:13,000)
    timestamp_pattern = re.compile(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$')

    # Loop through each line in the SRT data
    for line in srt_content:
        # Skip timestamp lines
        if timestamp_pattern.match(line.strip()):
            continue

        # Skip empty lines and lines that are just numbers (e.g., SRT entry index)
        if line.strip() == "" or line.strip().isdigit():
            continue

        # Add the speaker and dialogue to the final output
        dialogue += line.strip() + " "

    return dialogue.strip()  # Return the formatted dialogue as a string

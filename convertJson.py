import json

# Read JSON data from file
with open('output.json', 'r') as file:
    json_data = json.load(file)


def format_time(seconds):
    """Convert time in seconds to SRT time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def process_transcript(data):
    """Process JSON data to SRT format."""
    entries = []
    current_speaker = None
    current_start = None
    current_text = []

    for item in data['speakers']:
        if item['speaker'] == current_speaker:
            # Continue current speaker's text
            current_text.append(item['text'])
        else:
            if current_speaker is not None:
                # Save the current segment before switching speakers
                entries.append((current_speaker, current_start, item['timestamp'][0], " ".join(current_text)))
            # Start new segment
            current_speaker = item['speaker']
            current_start = item['timestamp'][0]
            current_text = [item['text']]

    # Don't forget to add the last segment
    if current_speaker is not None:
        entries.append((current_speaker, current_start, data['speakers'][-1]['timestamp'][1], " ".join(current_text)))

    # Format entries into SRT
    srt_entries = []
    for idx, entry in enumerate(entries, 1):
        start_time = format_time(entry[1])
        end_time = format_time(entry[2])
        speaker_text = f"{entry[0]}: {entry[3]}"
        srt_entries.append(f"{idx}\n{start_time} --> {end_time}\n{speaker_text}\n")

    return "\n".join(srt_entries)


# Convert to SRT format
srt_content = process_transcript(json_data)

# Write to SRT file
with open('output.srt', 'w') as srt_file:
    srt_file.write(srt_content)

print("SRT file has been created successfully.")

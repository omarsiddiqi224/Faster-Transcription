import subprocess

def run_cli():
    # Define the command as a string
    command = "insanely-fast-whisper --hf-token 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 --file-name audio.wav --flash True"

    command = "insanely-fast-whisper --hf-token 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 --file-name audio.wav"
    # Use subprocess to run the command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get the output and error messages, if any
    stdout, stderr = process.communicate()

    # Decode the output and error messages from bytes to string
    stdout = stdout.decode()
    stderr = stderr.decode()

    # Print the output and error messages
    print("Output:", stdout)
    if stderr:
        print("Error:", stderr)

# Call the function to run the CLI
#run_cli()

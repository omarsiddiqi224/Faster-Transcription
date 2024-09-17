# Use pytorch base image with CUDA 12.1
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Update the package manager and install pipx
RUN apt-get update && \
    apt-get install -y python3-pip pipx && \
    pipx ensurepath

# Install necessary Python dependencies
RUN python3 -m pip install --upgrade pip

# Clone the insanely-fast-whisper repository
RUN git clone https://github.com/Vaibhavs10/insanely-fast-whisper.git

# Set the working directory to the cloned repository
WORKDIR /workspace/insanely-fast-whisper

# Install requirements and the tool using pipx
RUN pipx install insanely-fast-whisper
RUN pip install -r requirements.txt

# Install torch, torchvision, and torchaudio for CUDA 12.1
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn with no build isolation using pipx
RUN pipx runpip insanely-fast-whisper install flash-attn --no-build-isolation

# Add entry point to run the appTest.py file
CMD ["python", "app.py"]

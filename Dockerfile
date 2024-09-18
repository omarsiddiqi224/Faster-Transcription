# Use pytorch base image with CUDA 12.1
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Set working directory
WORKDIR /workspace

# Update the package manager and install pipx
RUN apt-get update && \
    apt-get install -y python3-pip pipx git tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    pipx ensurepath

# Ensure pipx-installed binaries are in PATH
ENV PATH=$PATH:/root/.local/bin

# Install necessary Python dependencies
RUN python3 -m pip install --upgrade pip

# Clone the insanely-fast-whisper repository
RUN git clone https://github.com/Vaibhavs10/insanely-fast-whisper.git

# Set the working directory to the cloned repository
WORKDIR /workspace/insanely-fast-whisper

# Install requirements and the tool using pipx
RUN pipx install insanely-fast-whisper

# Set the working directory back to main directory
WORKDIR /workspace

# Copy the requirements.txt file and all necessary Python files to the Docker image
COPY requirements.txt /workspace/
COPY app.py /workspace/
COPY convert_srt.py /workspace/
COPY faster.py /workspace/
COPY convertJson.py /workspace/
COPY convert_text.py /workspace/

# Install torch, torchvision, and torchaudio for CUDA 12.1
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn with no build isolation using pipx
RUN pipx runpip insanely-fast-whisper install flash-attn --no-build-isolation

# Install the dependencies from the requirements.txt file
RUN pip install -r requirements.txt

# Add entry point to run the app.py file
CMD ["python", "app.py"]

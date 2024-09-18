# Faster-Transcription

This repository contains the Faster-Transcription project, which leverages state-of-the-art models for speech-to-text transcription. This guide provides steps to build and run the Docker image to use the transcription system.

## Getting Started

### Prerequisites

Make sure you have the following installed on your machine:
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

### Instructions

Follow these steps to get the project up and running:

1. **Clone the repository**
   
   Clone the `Faster-Transcription` repository to your local machine:

   ```bash
   git clone https://github.com/Vaibhavs10/insanely-fast-whisper.git
   ```

2. **Change directory**

   Navigate into the cloned `Faster-Transcription` directory:

   ```bash
   cd Faster-Transcription
   ```

3. **Build the Docker image**

   Build the Docker image using the provided Dockerfile. Replace `transcribe` with your preferred image name:

   ```bash
   docker build -t transcribe .
   ```

4. **Run the Docker container**

   Once the image is built, you can run the container. This command will run the container with GPU support. Replace `transcribe_container` with your preferred container name:

   ```bash
   docker run --gpus all -it --name transcribe_container transcribe
   ```

### Notes

- Ensure your machine has CUDA-enabled GPUs and that the NVIDIA Container Toolkit is installed for GPU acceleration.
- The Docker image uses Python and CUDA 12.1 for optimal performance.

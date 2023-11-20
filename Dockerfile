# Since we're using the nvidia/cuda base image, this requires nvidia-container-toolkit installed on the host system to pass through the drivers to the container.
# see: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04 AS final
WORKDIR /app

# Install Git and Git LFS
RUN apt-get update && apt-get install -y curl wget
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install -y git-lfs

# Clone the Stable Diffusion 1.5 base model
RUN git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 -b onnx

# Clone the LCM Dreamshaper V7 model
RUN git clone https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX

#need to install NVIDIA's gpg key before apt search will show up to date packages for cuda
RUN wget -N -t 5 -T 10 http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i ./cuda-keyring_1.1-1_all.deb

# install CUDA dependencies required according to `ldd libonnxruntime_providers_cuda.so`
RUN apt-get update \
    && apt-get install -y libcublaslt11 libcublas11 libcudnn8=8.9.1.23-1+cuda11.8 libcufft10 libcudart11.0

# According to `ldd libortextensions.so` it depends on ssl 1.1 to run, and the dotnet/runtime-deps base image installs it which is why it works inside the dotnet base images.
# Since we need access to the GPU to use the CUDA execution provider we need to use the nvidia/cuda base image instead.
# The nvidia/cuda base image doesn't contain SSL 1.1, hence we have to manually install it like this ot satisfy the dependency.
# This fixes the "The ONNX Runtime extensions library was not found" error.
# See: https://stackoverflow.com/questions/72133316/libssl-so-1-1-cannot-open-shared-object-file-no-such-file-or-directory
RUN wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.20_amd64.deb && dpkg -i libssl1.1_1.1.1f-1ubuntu2.20_amd64.deb

# Need to install dotnet sdk since we're not using the dotnet/sdk base image.
# Note: icu is also installed to help with globalization https://learn.microsoft.com/en-us/dotnet/core/extensions/globalization-icu
RUN apt-get update \
    && apt-get install -y dotnet-sdk-7.0 icu-devtools

ENV \
    # Enable detection of running in a container
    DOTNET_RUNNING_IN_CONTAINER=true \
    # Do not generate certificate
    DOTNET_GENERATE_ASPNET_CERTIFICATE=false \
    # Do not show first run text
    DOTNET_NOLOGO=true \
    # Skip extraction of XML docs - generally not useful within an image/container - helps performance
    NUGET_XMLDOC_MODE=skip

COPY . .
RUN dotnet build OnnxStackCore.sln

ENTRYPOINT ["sh", "-c", "nvidia-smi && dotnet test OnnxStackCore.sln"]
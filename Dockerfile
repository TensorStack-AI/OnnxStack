FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /app

# Install Git and Git LFS
RUN apt-get update && apt-get install -y curl
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install -y git-lfs

# Clone the Stable Diffusion 1.5 base model
RUN git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 -b onnx

COPY . .
RUN dotnet build OnnxStackCore.sln

ENTRYPOINT ["dotnet", "test", "OnnxStackCore.sln"]
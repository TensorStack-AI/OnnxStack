<p align="center" width="100%">
    <img width="80%" src="Assets/OnnxStack - 640x320.png">
</p>


![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/saddam213/OnnxStack)
[![Discord](https://img.shields.io/discord/1170119641545314375?label=Discord&)](https://discord.gg/YmUpFaaFsf)
[![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UCaWjlIC2uz8hDIZCBXjyDRw)](https://www.youtube.com/@OnnxStack)
### **[Discord](https://discord.gg/YxbQfbegdS)**  | **[Youtube](https://www.youtube.com/@OnnxStack)**
__________________________




# Welcome to OnnxStack!
OnnxStack transforms machine learning in .NET, Seamlessly integrating with `ONNX Runtime` and `Microsoft ML`, this library empowers you to build, deploy, and execute machine learning models entirely within the .NET ecosystem. Bid farewell to Python dependencies and embrace a new era of intelligent applications tailored for .NET

### Welcome to Python-free AI integration with OnnxStack!




# Projects

## **[OnnxStack.Core](OnnxStack.Core/README.md)**
**Model Inference with C# and ONNX Runtime**

[![Core Badge](https://img.shields.io/nuget/v/OnnxStack.Core?color=4bc51e&label=OnnxStack.Core)](https://www.nuget.org/packages/OnnxStack.Core)
![Nuget](https://img.shields.io/nuget/dt/OnnxStack.Core)

`OnnxStack.Core` is a .NET library designed to facilitate seamless interaction with the `OnnxRuntime` C# API. This project simplifies the creation and disposal of `OrtValues` and offers straightforward services for loading and running inferences on a variety of models. With a focus on improving developer efficiency, the library abstracts complexities, allowing for smoother integration of `OnnxRuntime` into .NET applications.

More information and examples can be found in the `OnnxStack.Core` project **[README](OnnxStack.Core/README.md)**

__________________________

__________________________


## **[OnnxStack.StableDiffusion](OnnxStack.StableDiffusion/README.md)**
**Stable Diffusion Inference with C# and ONNX Runtime**

[![StableDiffusion Badge](https://img.shields.io/nuget/v/OnnxStack.StableDiffusion?color=4bc51e&label=OnnxStack.StableDiffusion)](https://www.nuget.org/packages/OnnxStack.StableDiffusion)
![Nuget](https://img.shields.io/nuget/dt/OnnxStack.StableDiffusion)

`OnnxStack.StableDiffusion` is a .NET library for latent diffusion in C#, Leveraging `OnnxStack.Core`, this library seamlessly integrates many StableDiffusion capabilities, including:
* Text to Image
* Image to Image
* Image Inpaint
* Video to Video
* Control Net


`OnnxStack.StableDiffusion` provides compatibility with a diverse set of models, including 
* StableDiffusion 1.5
* StableDiffusion Inpaint
* StableDiffusion ControlNet
* Stable-Cascade
* SDXL
* SDXL Inpaint
* SDXL-Turbo
* LatentConsistency
* LatentConsistency XL
* Instaflow


More information can be found in the `OnnxStack.StableDiffusion` project **[README](OnnxStack.StableDiffusion/README.md)**

__________________________
__________________________
## **[OnnxStack.ImageUpscaler]()**
**Image upscaler with C# and ONNX Runtime**

[![Upscale Badge](https://img.shields.io/nuget/v/OnnxStack.ImageUpscaler?color=4bc51e&label=OnnxStack.ImageUpscaler)](https://www.nuget.org/packages/OnnxStack.ImageUpscaler)
![Nuget](https://img.shields.io/nuget/dt/OnnxStack.ImageUpscaler)

`OnnxStack.ImageUpscaler` is a library designed to elevate image quality through superior upscaling techniques. Leveraging `OnnxStack.Core`, this library provides seamless integration for enhancing image resolution and supports a variety of upscaling models, allowing developers to improve image clarity and quality. Whether you are working on image processing, content creation, or any application requiring enhanced visuals, the ImageUpscale project delivers efficient and high-quality upscaling solutions.


More information and examples can be found in the `OnnxStack.ImageUpscaler` project **[README](OnnxStack.ImageUpscaler/README.md)**
__________________________
__________________________
## **[OnnxStack.ImageRecognition]()**
**Image recognition with ResNet50v2 and ONNX Runtime**

Harness the accuracy of the ResNet50v2 deep learning model for image recognition, seamlessly integrated with ONNX for efficient deployment. This combination empowers your applications to classify images with precision, making it ideal for tasks like object detection, content filtering, and image tagging across various platforms and hardware accelerators. Achieve high-quality image recognition effortlessly with ResNet50v2 and ONNX integration.


***work in progress***
__________________________


__________________________
## **[OnnxStack.ObjectDetection]()**
**Object detection with Faster RCNN Deep Learning with C# and ONNX Runtime**

Enable robust object detection in your applications using RCNN (Region-based Convolutional Neural Network) integrated with ONNX. This powerful combination allows you to accurately locate and classify objects within images. Whether for surveillance, autonomous vehicles, or content analysis, RCNN and ONNX integration offers efficient and precise object detection across various platforms and hardware, ensuring your solutions excel in recognizing and localizing objects in images.

***work in progress***
__________________________


## Contribution

We welcome contributions to OnnxStack! If you have any ideas, bug reports, or improvements, feel free to open an issue or submit a pull request.

* Join our Discord: **[OnnxStack Discord](https://discord.gg/uQzQgxMYWy)** 
* Chat to us here: **[Project Discussion Board](https://github.com/saddam213/OnnxStack/discussions)** 



__________________________
## ONNX Runtime Resources
- [ONNX Runtime C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)

## Reference
Special thanks to the creators of the fantastic repositories below; all were instrumental in the creation of OnnxStack.

* [Stable Diffusion with C# and ONNX Runtime](https://github.com/cassiebreviu/stablediffusion) by Cassie Breviu (@cassiebreviu)
* [Diffusers](https://github.com/huggingface/diffusers) by Huggingface (@huggingface)
* [Onnx-Web](https://github.com/ssube/onnx-web) by Sean Sube (@ssube)
* [Axodox-MachineLearning](https://github.com/axodox/axodox-machinelearning) by Péter Major @(axodox) 
* [ControlNet](https://github.com/lllyasviel/ControlNet) by Lvmin Zhang (@lllyasviel)
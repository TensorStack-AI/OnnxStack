<p align="center" width="100%">
    <img width="80%" src="Assets/OnnxStack - 640x320.png">
</p>



[![Core Badge](https://img.shields.io/nuget/v/OnnxStack.Core?color=4bc51e&label=OnnxStack.Core)](https://www.nuget.org/packages/OnnxStack.Core)
![Nuget](https://img.shields.io/nuget/dt/OnnxStack.Core)
[![StableDiffusion Badge](https://img.shields.io/nuget/v/OnnxStack.StableDiffusion?color=4bc51e&label=OnnxStack.StableDiffusion)](https://www.nuget.org/packages/OnnxStack.StableDiffusion)
![Nuget](https://img.shields.io/nuget/dt/OnnxStack.StableDiffusion)
[![StableDiffusion Badge](https://img.shields.io/nuget/v/OnnxStack.StableDiffusion?color=4bc51e&label=OnnxStack.UI)](https://github.com/saddam213/OnnxStack/releases)
![Nuget](https://img.shields.io/github/downloads/saddam213/OnnxStack/total)


![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/saddam213/OnnxStack)
[![Discord](https://img.shields.io/discord/1170119641545314375?label=Discord&)](https://discord.gg/YmUpFaaFsf)
[![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UCaWjlIC2uz8hDIZCBXjyDRw)](https://www.youtube.com/@OnnxStack)





### Welcome to OnnxStack!
OnnxStack transforms machine learning in .NET, Seamlessly integrating with `ONNX Runtime` and `Microsoft ML`, this library empowers you to build, deploy, and execute machine learning models entirely within the .NET ecosystem. Bid farewell to Python dependencies and embrace a new era of intelligent applications tailored for .NET
Welcome to Python-free AI integration with OnnxStack!


## **Latest Release Download: [OnnxStack UI v0.10.0](https://github.com/saddam213/OnnxStack/releases/download/v0.10.0/WindowsUI.v0.10.0.zip)**

### **[Discord](https://discord.gg/YxbQfbegdS)**  | **[Youtube](https://www.youtube.com/@OnnxStack)**

# Projects

## **[OnnxStack.StableDiffusion](OnnxStack.StableDiffusion/README.md)**
Inference Stable Diffusion with C# and ONNX Runtime

###  **Prompt**

Stable Diffusion models take a text prompt and create an image that represents the text.

*Example:*
`
High-fashion photography in an abandoned industrial warehouse, with dramatic lighting and edgy outfits, detailed clothing, intricate clothing, seductive pose, action pose, motion, beautiful digital artwork, atmospheric, warm sunlight, photography, neo noir, bokeh, beautiful dramatic lighting, shallow depth of field, photorealism, volumetric lighting, Ultra HD, raytracing, studio quality, octane render
`

###  **Negative Prompt**

A negative prompt can be provided to guide the inference to exclude in calculations

*Example:*

`
painting, drawing, sketches, monochrome, grayscale, illustration, anime, cartoon, graphic, text, crayon, graphite, abstract, easynegative, low quality, normal quality, worst quality, lowres, close up, cropped, out of frame, jpeg artifacts, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, glitch, deformed, mutated, cross-eyed, ugly, dehydrated, bad anatomy, bad proportions, gross proportions, cloned face, disfigured, malformed limbs, missing arms, missing legs fused fingers, too many fingers,extra fingers, extra limbs,, extra arms, extra legs,disfigured,
`

### **Schedulers**

Many different scheduler algorithms can be used for this computation, each having its pro- and cons. 
So far `OnnxStack.StableDiffusion` as included `LMS Discrete`, `Euler Ancestral`, `DDPM`, `DDIM`, and `KDPM2` options with more in the works.

*Example:*
| LMS Scheduler | Euler Ancestral Scheduler | DDPM Scheduler |
| :--- | :--- | :--- |
<img src="Assets/Samples/624461087_22_8_LMSScheduler.png" width="256" alt="Image of browser inferencing on sample images."/> | <img src="Assets/Samples/624461087_22_8_EulerAncestralScheduler.png" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="Assets/Samples/624461087_22_8_DDPMScheduler.png" width="256"  alt="Image of browser inferencing on sample images."/> |

     Seed: 624461087     GuidanceScale: 8     NumInferenceSteps: 22

### **Text To Image**
Text To Image Stable Diffusion is a powerful machine learning technique that allows you to generate high-quality images from textual descriptions. It combines the capabilities of text understanding and image synthesis to convert natural language descriptions into visually coherent and meaningful images

| Input Text | Output Image | Diffusion Steps |
| :--- | :--- | :--- |
<img src="Assets/Samples/Text2Img_Start.png" width="256" alt="Image of browser inferencing on sample images."/> | <img src="Assets/Samples/Text2Img_End.png" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="Assets/Samples/Text2Img_Animation.webp" width="256"  alt="Image of browser inferencing on sample images."/> |

### **Image To Image**
Image To Image Stable Diffusion is an advanced image processing and generation method that excels in transforming one image into another while preserving the visual quality and structure of the original content. Using stable diffusion, this technique can perform a wide range of image-to-image tasks, such as style transfer, super-resolution, colorization, and more

| Input Image | Output Image | Diffusion Steps |
| :--- | :--- | :--- |
<img src="Assets/Samples/Img2Img_Start.bmp" width="256" alt="Image of browser inferencing on sample images."/> | <img src="Assets/Samples/Img2Img_End.png" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="Assets/Samples/Img2Img_Animation.webp" width="256"  alt="Image of browser inferencing on sample images."/> |

```
   Prompt: Dog wearing storm trooper helmet, head shot
```

### **Image Inpainting**
Image inpainting is an image modification/restoration technique that intelligently fills in missing or damaged portions of an image while maintaining visual consistency. It's used for tasks like photo restoration and object removal, creating seamless and convincing results.

In the below example we use a simple mask image + prompt to add a rider to the horse
The black part of the mask will be used buy the process to generate new content, in this case the rider

| Input Image | Mask Image | Masked Image | Result
| :--- | :--- | :--- | :--- |
<img src="Assets/Samples/Inpaint-Original.png" width="256" alt="Image of browser inferencing on sample images."/> | <img src="Assets/Samples/Inpaint-Mask.png" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="Assets/Samples/Inpaint-MaskedImage.PNG" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="Assets/Samples/Inpaint-Result.png" width="256"  alt="Image of browser inferencing on sample images."/> |

```
   Prompt: Rider on horse
```


## **Realtime Stable Diffusion**
Realtime stable diffusion is a process where the results are constantly rendered as you are working with the image or changing the settings, This can be fantastic if you are creating new artworks or editing existing images.

Performance will depend on hardware and models selected, but for `Latent Consistency Models` you can get up to 4fps with a 3090 :)

### Text To Image
https://user-images.githubusercontent.com/4353684/285347887-99db7f37-cff4-48b6-805b-3ca55e8f0c3a.mp4

### Image To Image
https://user-images.githubusercontent.com/4353684/285348410-c19a2111-6745-4f01-8400-d137d40180fe.mp4

### Image Inpaint 
https://user-images.githubusercontent.com/4353684/285347894-9d044d7d-7c22-4379-8187-9cf7b9cac89c.mp4

### Paint To Image 
https://user-images.githubusercontent.com/4353684/285347896-8da6709b-fea6-4cd4-ba65-ec692401f475.mp4

https://user-images.githubusercontent.com/4353684/285547207-3a7ea067-fcbf-47f0-9372-fafa94d301f7.mp4



More information and Examples can be found in the `OnnxStack.StableDiffusion` project **[README](OnnxStack.StableDiffusion/README.md)**

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


# Getting Started


## ONNX Model Download
You will need an ONNX compatible model to use, Hugging Face is a great place to download the Stable Diffusion models

Download the [ONNX Stable Diffusion models from Hugging Face](https://huggingface.co/models?sort=downloads&search=Stable+Diffusion).

- [Stable Diffusion Models v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/onnx)
- [Stable Diffusion Models v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/onnx)


Once you have selected a model version repo, click `Files and Versions`, then select the `ONNX` branch. If there isn't an ONNX model branch available, use the `main` branch and convert it to ONNX. See the [ONNX conversion tutorial for PyTorch](https://learn.microsoft.com/windows/ai/windows-ml/tutorials/pytorch-convert-model) for more information.

Clone the model repo:
```text
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 -b onnx
```


## Stable Diffusion Models We've Converted To ONNX And Tested
The following are othe stable diffusion models we've converted to ONNX and tested with OnnxStack.

- [OpenJourney V4](https://huggingface.co/TheyCallMeHex/OpenJourney-V4-ONNX)
- [DreamLike PhotoReal 2.0](https://huggingface.co/TheyCallMeHex/DreamLike-PhotoReal-2.0-ONNX)
- [CyberPunk Anime Diffusion](https://huggingface.co/TheyCallMeHex/Cyberpunk-Anime-Diffusion-ONNX)
- [InkPunk Diffusion](https://huggingface.co/TheyCallMeHex/Inkpunk-Diffusion-ONNX)
- [Mo-Di Diffusion](https://huggingface.co/TheyCallMeHex/Mo-Di-Diffusion-ONNX)
- [epiCRealism](https://huggingface.co/TheyCallMeHex/epiCRealism-ONNX)
- [Comic Diffusion](https://huggingface.co/TheyCallMeHex/Comic-Diffusion-ONNX)
- [Redshift Diffusion](https://huggingface.co/TheyCallMeHex/Redshift-Diffusion-ONNX)


## Hardware Requirements
You can choose between `Cpu` and `DirectML`(GPU) for inference, 
Other `Microsoft.ML.OnnxRuntime.*` executors like `Cuda` may work but are untested

`Cpu` > 12GB RAM

`DirectML` > 10GB VRAM

## Contribution

We welcome contributions to OnnxStack! If you have any ideas, bug reports, or improvements, feel free to open an issue or submit a pull request.



__________________________
##  Resources
- [ONNX Runtime C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
- [Hugging Face Stable Diffusion Blog](https://huggingface.co/blog/stable_diffusion)
- [Original C# Stable Diffusion blog](https://devblogs.microsoft.com/dotnet/generate-ai-images-stable-diffusion-csharp-onnx-runtime/)
- [Original ONNX Runtime tutorial for Stable Diffusion in C#](https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html)

## Reference

This work is based on the original C# implementation of Stable Diffusion by Cassie Breviu here: [Stable Diffusion with C# and ONNX Runtime](https://github.com/cassiebreviu/stablediffusion).
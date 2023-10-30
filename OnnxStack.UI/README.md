<p align="center" width="100%">
    <img width="80%" src="Assets/OnnxStack - 640x320.png">
</p>


![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/saddam213/OnnxStack)
[![Core Badge](https://img.shields.io/nuget/v/OnnxStack.Core?color=4bc51e&label=OnnxStack.Core)](https://www.nuget.org/packages/OnnxStack.Core)
![Nuget](https://img.shields.io/nuget/dt/OnnxStack.Core)
[![StableDiffusion Badge](https://img.shields.io/nuget/v/OnnxStack.StableDiffusion?color=4bc51e&label=OnnxStack.StableDiffusion)](https://www.nuget.org/packages/OnnxStack.StableDiffusion)
![Nuget](https://img.shields.io/nuget/dt/OnnxStack.StableDiffusion)



### Welcome to OnnxStack WPF GUI!
This project is a sample WPF GUI of the OnnxStack library's Inference Stable Diffusion implementation.

###  **Prompt**

Stable Diffusion models take a text prompt and create an image that represents the text.

*Example:*
`
final fantasy style cyberpunk desert assassin girl with thigh-high boot
`

###  **Negative Prompt**

A negative prompt can be provided to guide the inference to exclude in calculations

*Example:*

`
painting, drawing, sketches, monochrome, grayscale, illustration, anime, cartoon, graphic, text, crayon, graphite, abstract, easynegative, low quality, normal quality, worst quality, lowres, close up, cropped, out of frame, jpeg artifacts, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, glitch, deformed, mutated, cross-eyed, ugly, dehydrated, bad anatomy, bad proportions, gross proportions, cloned face, disfigured, malformed limbs, missing arms, missing legs fused fingers, too many fingers,extra fingers, extra limbs,, extra arms, extra legs,disfigured,
`

### **Schedulers**

Many different scheduler algorithms can be used for this computation, each having its pro- and cons. 
So far `OnnxStack.StableDiffusion` as included `LMS Discrete`, `Euler Ancestral` and `DDPM` and `DDIM` options with more in the works.

*Example:*
| LMS Scheduler | Euler Ancestral Scheduler | DDPM Scheduler | DDIM Scheduler
| :--- | :--- | :--- | :--- |
<img src="../Assets/Samples/1207582124_30_7.5_30_LMS.png" width="256" alt="Image of browser inferencing on sample images."/> | <img src="../Assets/Samples/1207582124_30_7.5_30_EulerAncestral.png" width="256"  alt="Image of browser inferencing on sample images."/> | <img src="../Assets/Samples/1207582124_30_7.5_30_DDPM.png" width="256"  alt="Image of browser inferencing on sample images."/> | <img src="../Assets/Samples/1207582124_30_7.5_30_DDIM.png" width="256"  alt="Image of browser inferencing on sample images."/> |

     Model: OpenJourney V4     Seed: A cyberpunk puppy     GuidanceScale: 7.5     NumInferenceSteps: 30     Prompt: A cyberpunk puppy

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
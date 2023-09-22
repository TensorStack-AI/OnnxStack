# OnnxStack
ONNX Runtime Projects for .NET Applications

## Hardware Requirements
You can choose between `Cpu` and `DirectML`(GPU) for inference, 
Other `Microsoft.ML.OnnxRuntime.*` executors like `Cuda` may work but are untested

`Cpu` > 12GB RAM

`DirectML` > 10GB VRAM


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


# Projects

## `OnnxStack.StableDiffusion`
Inference Stable Diffusion with C# and ONNX Runtime

Stable Diffusion models take a text prompt and create an image that represents the text.

Prompt Example:
```text
"Renaissance-style portrait of an astronaut in space, detailed starry background, reflective helmet." 
```

Many different scheduler algorithms can be used for this computation, each having its pro- and cons. 
So far `OnnxStack.StableDiffusion` as included `LMSScheduler` and `EulerAncestralScheduler` options with more in the works.


Scheduler Output Examples:
| LMSScheduler | EulerAncestralScheduler|
| :--- | :--- |
<img src="https://i.imgur.com/Ptaai09.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> | <img src="https://i.imgur.com/6nZNi7A.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> |

__________________________


__________________________
## `OnnxStack.ImageRecognition`
Image recognition with ResNet50v2 with C# and ONNX Runtime

~WIP~
__________________________


__________________________
## `OnnxStack.ObjectDetection`
Object detection with Faster RCNN Deep Learning with C# and ONNX Runtime

~WIP~
__________________________


## Contribution

We welcome contributions to OnnxStack! If you have any ideas, bug reports, or improvements, feel free to open an issue or submit a pull request.



__________________________
##  Resources
- [ONNX Runtime C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
- [Hugging Face Stable Diffusion Blog](https://huggingface.co/blog/stable_diffusion)
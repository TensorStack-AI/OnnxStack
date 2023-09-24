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

## **[OnnxStack.StableDiffusion](OnnxStack.StableDiffusion/README.md)**
Inference Stable Diffusion with C# and ONNX Runtime

**`Prompt`**

Stable Diffusion models take a text prompt and create an image that represents the text.

*Example:*
```text
High-fashion photography in an abandoned industrial warehouse, with dramatic lighting and edgy outfits, detailed clothing, intricate clothing, seductive pose, action pose, motion, beautiful digital artwork, atmospheric, warm sunlight, photography, neo noir, bokeh, beautiful dramatic lighting, shallow depth of field, photorealism, volumetric lighting, Ultra HD, raytracing, studio quality, octane render
```

**`Negative Prompt`**

A negative prompt can be provided to guide the inference to exclude in calculations

*Example:*
```text
painting, drawing, sketches, monochrome, grayscale, illustration, anime, cartoon, graphic, text, crayon, graphite, abstract, easynegative, low quality, normal quality, worst quality, lowres, close up, cropped, out of frame, jpeg artifacts, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, glitch, deformed, mutated, cross-eyed, ugly, dehydrated, bad anatomy, bad proportions, gross proportions, cloned face, disfigured, malformed limbs, missing arms, missing legs fused fingers, too many fingers,extra fingers, extra limbs,, extra arms, extra legs,disfigured,
```

**`Schedulers`**

Many different scheduler algorithms can be used for this computation, each having its pro- and cons. 
So far `OnnxStack.StableDiffusion` as included `LMSScheduler` and `EulerAncestralScheduler` options with more in the works.

*Example:*
| LMSScheduler | EulerAncestralScheduler|
| :--- | :--- |
<img src="https://i.imgur.com/plsRi2K.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> | <img src="https://i.imgur.com/mDxIEg2.png" width="256" height="256" alt="Image of browser inferencing on sample images."/> |

     Seed: 393371621     GuidanceScale: 8     NumInferenceSteps: 30



More information and Examples can be found in the `OnnxStack.StableDiffusion` project **[README](OnnxStack.StableDiffusion/README.md)**

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
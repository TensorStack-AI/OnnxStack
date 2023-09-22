# OnnxStack
ONNX Runtime Projects for .NET Applications

## Hardware Requirements
You can choose between `Cpu` and `DirectML`(GPU) for inference, 
Other `Microsoft.ML.OnnxRuntime.*` executors like `Cuda` may work but are untested

`Cpu` > 12GB RAM

`DirectML` > 10GB VRAM


## ONNX Model Download
You will need an ONNX compatable model to use, Hugging Face is a greaty place to download the Stable Diffusion models

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

## OnnxStack.StableDiffusion
Inference Stable Diffusion with C# and ONNX Runtime


### Basic C# Example of using OnnxStack.StableDiffusion
```csharp

// Create Configuration
var onnxStackConfig = new OnnxStackConfig
{
    IsSafetyModelEnabled = false,
    ExecutionProviderTarget = ExecutionProvider.DirectML,
    OnnxUnetPath = "stable-diffusion-v1-5\\unet\\model.onnx",
    OnnxVaeDecoderPath = "stable-diffusion-v1-5\\vae_decoder\\model.onnx",
    OnnxTextEncoderPath = "stable-diffusion-v1-5\\text_encoder\\model.onnx",
    OnnxSafetyModelPath = "stable-diffusion-v1-5\\safety_checker\\model.onnx"
};

// Create Service
using (var stableDiffusionService = new StableDiffusionService(onnxStackConfig))
{
    // StableDiffusion Options
    var options = new StableDiffusionOptions
    {
        Prompt = "Renaissance-style portrait of an astronaut in space, detailed starry background, reflective helmet",
        NegativePrompt = "planets, moon",
        GuidanceScale = 7.5,
        NumInferenceSteps = 30,
        SchedulerType = SchedulerType.LMSScheduler
    };

    // 1. Create an image in memory
    var image = await stableDiffusionService.TextToImage(options)

    //OR

    // 2. Create image and save to disk
    var outputFilename = "MyImage.png"
    var success = await stableDiffusionService.TextToImageFile(options, outputFilename)
}
```






__________________________
##  Resources
- [ONNX Runtime C# API Doc](https://onnxruntime.ai/docs/api/csharp/api)
- [Get Started with C# in ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html)
- [Hugging Face Stable Diffusion Blog](https://huggingface.co/blog/stable_diffusion)
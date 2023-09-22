# OnnxStack
C# Stable Diffusion using ONNX Runtime


```csharp

// Create Configuration
var onnxStackConfig = new OnnxStackConfig
{
    ExecutionProviderTarget = ExecutionProvider.DirectML,
    OnnxUnetPath = "stable-diffusion-v1-5\\unet\\model.onnx",
    OnnxVaeDecoderPath = "stable-diffusion-v1-5\\vae_decoder\\model.onnx",
    OnnxTextEncoderPath = "stable-diffusion-v1-5\\text_encoder\\model.onnx"
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

``

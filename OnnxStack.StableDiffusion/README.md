# OnnxStack.StableDiffusion - Onnx Stable Diffusion Library for .NET

`OnnxStack.StableDiffusion` is a library that provides access to Stable Diffusion processes in .NET. 
It offers extensive support for features such as TextToImage, ImageToImage, VideoToVideo, ControlNet and more

## Getting Started

OnnxStack.StableDiffusion can be found via the nuget package manager, download and install it.
```
PM> Install-Package OnnxStack.StableDiffusion
```

## OnnxRuntime
Depending on the devices you have and the platform you are running on, you will want to install the `Microsoft.ML.OnnxRuntime` package that best suits your needs.

**DirectML** - CPU-GPU support for Windows (Windows)
```
PM> Install-Package Microsoft.ML.OnnxRuntime.DirectML
```

**CUDA** - GPU support for NVIDIA (Windows, Linux)
```
PM> Install-Package Microsoft.ML.OnnxRuntime.Gpu
```

**CoreML** - CPU-GPU support for Mac (Apple)
```
PM> Install-Package Microsoft.ML.OnnxRuntime.CoreML
```


## Dependencies
Video processing support requires `FFMpeg` and `FFProbe` binaries, files must be present in your output folder
```
https://ffbinaries.com/downloads
https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffmpeg-6.1-win-64.zip
https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffprobe-6.1-win-64.zip
```


# C# Stable Diffusion
Example Model: https://huggingface.co/runwayml/stable-diffusion-v1-5 (onnx branch)


## Basic Stable Diffusion Example
Run a simple Stable Diffusion process with a basic prompt
```csharp
// Create Pipeline
var pipeline = StableDiffusionPipeline.CreatePipeline("D:\\Repositories\\stable-diffusion-v1-5");

// Set Prompt Options
var promptOptions = new PromptOptions { Prompt = "Photo of a cute dog." };

// Run Pipleine
var result = await pipeline.RunAsync(promptOptions);

// Save image result
var image = result.ToImage();
await image.SaveAsPngAsync("D:\\Results\\Image.png");

// Unload Pipleine
await pipeline.UnloadAsync();
```

## Stable Diffusion Batch Example
Run Stable Diffusion process and return a batch of results
```csharp
// Create Pipeline
var pipeline = StableDiffusionPipeline.CreatePipeline("D:\\Repositories\\stable-diffusion-v1-5");

// Prompt
var promptOptions = new PromptOptions{ Prompt = "Photo of a cat" };

// Batch Of 5 Images with unique seeds
var batchOptions = new BatchOptions
{
    ValueTo = 5,
    BatchType = BatchOptionType.Seed
};

// Run Pipleine
await foreach (var result in pipeline.RunBatchAsync(batchOptions, promptOptions))
{
    // Save Image result
   var image = result.ImageResult.ToImage();
   await image.SaveAsPngAsync($"D:\\Results\\Image_{result.SchedulerOptions.Seed}.png");
}

// Unload Pipleine
await pipeline.UnloadAsync();

```

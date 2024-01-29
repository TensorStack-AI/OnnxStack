# OnnxStack.StableDiffusion - Onnx Stable Diffusion Services for .NET Applications

`OnnxStack.StableDiffusion` is a library that provides higher-level Stable Diffusion services for use in .NET applications. It offers extensive support for features such as dependency injection, .NET configuration implementations, ASP.NET Core integration, and IHostedService support.

## Getting Started

OnnxStack.StableDiffusion can be found via the nuget package manager, download and install it.
```
PM> Install-Package OnnxStack.StableDiffusion
```

### Microsoft.ML.OnnxRuntime
Depending on the devices you have and the platform you are running on, you will want to install the Microsoft.ML.OnnxRuntime package that best suits your needs.

### CPU-GPU via Microsoft Drirect ML
```
PM> Install-Package Microsoft.ML.OnnxRuntime.DirectML
```

### GPU support for NVIDIA
```
PM> Install-Package Microsoft.ML.OnnxRuntime.Gpu
```



### .NET Core Registration

You can easily integrate `OnnxStack.StableDiffusion` into your application services layer. This registration process sets up the necessary services and loads the `appsettings.json` configuration.

Example: Registering OnnxStack.StableDiffusion
```csharp
builder.Services.AddOnnxStackStableDiffusion();
```

## Dependencies
Video processing support requires FFMPEG and FFPROBE binaries, files must be present in your output folder or the destinations configured in the `appsettings.json`
```
https://ffbinaries.com/downloads
https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffmpeg-6.1-win-64.zip
https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffprobe-6.1-win-64.zip
```


# C# Stable Diffusion Examples
Excample Model: https://huggingface.co/runwayml/stable-diffusion-v1-5 (onnx branch)


## Basic Stable Diffusion
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




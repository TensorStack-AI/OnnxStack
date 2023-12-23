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

### GPU support for both NVIDIA and AMD?
```
PM> Install-Package Microsoft.ML.OnnxRuntime.Gpu
```



### .NET Core Registration

You can easily integrate `OnnxStack.StableDiffusion` into your application services layer. This registration process sets up the necessary services and loads the `appsettings.json` configuration.

Example: Registering OnnxStack.StableDiffusion
```csharp
builder.Services.AddOnnxStackStableDiffusion();
```




## .NET Console Application Example

Required Nuget Packages for example
```nuget
Microsoft.Extensions.Hosting
Microsoft.Extensions.Logging
```

```csharp
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;

internal class Program
{
   static async Task Main(string[] _)
   {
      var builder = Host.CreateApplicationBuilder();
      builder.Logging.ClearProviders();
      builder.Services.AddLogging((loggingBuilder) => loggingBuilder.SetMinimumLevel(LogLevel.Error));

      // Add OnnxStack Stable Diffusion
      builder.Services.AddOnnxStackStableDiffusion();

      // Add AppService
      builder.Services.AddHostedService<AppService>();

      // Start
      await builder.Build().RunAsync();
   }
}

internal class AppService : IHostedService
{
   private readonly string _outputDirectory;
   private readonly IStableDiffusionService _stableDiffusionService;

   public AppService(IStableDiffusionService stableDiffusionService)
   {
      _stableDiffusionService = stableDiffusionService;
      _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Images");
   }

   public async Task StartAsync(CancellationToken cancellationToken)
   {
      Directory.CreateDirectory(_outputDirectory);

      while (true)
      {
         System.Console.WriteLine("Please type a prompt and press ENTER");
         var prompt = System.Console.ReadLine();

         System.Console.WriteLine("Please type a negative prompt and press ENTER (optional)");
         var negativePrompt = System.Console.ReadLine();


         // Example only, full config depends on model
         // appsettings.json is recommended for ease of use
         var modelOptions = new ModelOptions
         {
            Name = "Stable  Diffusion 1.5",
            ExecutionProvider = ExecutionProvider.DirectML,
            ModelConfigurations = new List<OnnxModelSessionConfig>
            {
                  new OnnxModelSessionConfig
                  {
                     Type = OnnxModelType.Unet,
                     OnnxModelPath = "model path"
                  }
            }
         };

         var promptOptions = new PromptOptions
         {
            Prompt = prompt,
            NegativePrompt = negativePrompt,
            DiffuserType = DiffuserType.TextToImage,

            // Input for ImageToImage
            // InputImage = new InputImage(File.ReadAllBytesAsync("image to image filename"))
         };

         var schedulerOptions = new SchedulerOptions
         {
            Seed = Random.Shared.Next(),
            GuidanceScale = 7.5f,
            InferenceSteps = 30,
            Height = 512,
            Width = 512,
            SchedulerType = SchedulerType.LMS,
         };


         // Generate Image Example
         var outputFilename = Path.Combine(_outputDirectory, $"{schedulerOptions.Seed}_{schedulerOptions.SchedulerType}.png");
         var result = await _stableDiffusionService.GenerateAsImageAsync(modelOptions, promptOptions, schedulerOptions);
         if (result is not null)
         {
            // Save image to disk
            await result.SaveAsPngAsync(outputFilename);
         }




         // Generate Batch Example
         var batchOptions = new BatchOptions
         {
            BatchType = BatchOptionType.Seed,
            ValueTo = 20
         };

         await foreach (var batchResult in _stableDiffusionService.GenerateBatchAsImageAsync(modelOptions, promptOptions, schedulerOptions, batchOptions))
         {
            // Save image to disk
            await batchResult.SaveAsPngAsync(outputFilename);
         }


      }
   }

   public Task StopAsync(CancellationToken cancellationToken)
   {
      return Task.CompletedTask;
   }
}
```


## Configuration
The `appsettings.json` is the easiest option for configuring model sets. Below is an example of `Stable Diffusion 1.5`.
The example adds the necessary paths to each model file required for Stable Diffusion, as well as any model-specific configurations. 
Each model can be assigned to its own device, which is handy if you have only a small GPU. This way, you can offload only what you need. There are limitations depending on the version of the `Microsoft.ML.OnnxRuntime` package you are using, but in most cases, you can split the load between CPU and GPU.

```json
{
   "Logging": {
      "LogLevel": {
         "Default": "Information",
         "Microsoft.AspNetCore": "Warning"
      }
   },

   "OnnxStackConfig": {
      "Name": "StableDiffusion 1.5",
      "IsEnabled": true,
      "PadTokenId": 49407,
      "BlankTokenId": 49407,
      "TokenizerLimit": 77,
      "EmbeddingsLength": 768,
      "ScaleFactor": 0.18215,
      "PipelineType": "StableDiffusion",
      "Diffusers": [
         "TextToImage",
         "ImageToImage",
         "ImageInpaintLegacy"
      ],
      "DeviceId": 0,
      "InterOpNumThreads": 0,
      "IntraOpNumThreads": 0,
      "ExecutionMode": "ORT_SEQUENTIAL",
      "ExecutionProvider": "DirectML",
      "ModelConfigurations": [
         {
            "Type": "Tokenizer",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\cliptokenizer.onnx"
         },
         {
            "Type": "Unet",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\unet\\model.onnx"
         },
         {
            "Type": "TextEncoder",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\text_encoder\\model.onnx"
         },
         {
            "Type": "VaeEncoder",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\vae_encoder\\model.onnx"
         },
         {
            "Type": "VaeDecoder",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\vae_decoder\\model.onnx"
         }
      ]
   }
}
```

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
<img src="../Assets/Samples/624461087_22_8_LMSScheduler.png" width="256" alt="Image of browser inferencing on sample images."/> | <img src="../Assets/Samples/624461087_22_8_EulerAncestralScheduler.png" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="../Assets/Samples/624461087_22_8_DDPMScheduler.png" width="256"  alt="Image of browser inferencing on sample images."/> |

     Seed: 624461087     GuidanceScale: 8     NumInferenceSteps: 22

### **Text To Image**
Text To Image Stable Diffusion is a powerful machine learning technique that allows you to generate high-quality images from textual descriptions. It combines the capabilities of text understanding and image synthesis to convert natural language descriptions into visually coherent and meaningful images

| Input Text | Output Image | Diffusion Steps |
| :--- | :--- | :--- |
<img src="../Assets/Samples/Text2Img_Start.png" width="256" alt="Image of browser inferencing on sample images."/> | <img src="../Assets/Samples/Text2Img_End.png" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="../Assets/Samples/Text2Img_Animation.webp" width="256"  alt="Image of browser inferencing on sample images."/> |

### **Image To Image**
Image To Image Stable Diffusion is an advanced image processing and generation method that excels in transforming one image into another while preserving the visual quality and structure of the original content. Using stable diffusion, this technique can perform a wide range of image-to-image tasks, such as style transfer, super-resolution, colorization, and more

| Input Image | Output Image | Diffusion Steps |
| :--- | :--- | :--- |
<img src="../Assets/Samples/Img2Img_Start.bmp" width="256" alt="Image of browser inferencing on sample images."/> | <img src="../Assets/Samples/Img2Img_End.png" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="../../Assets/Samples/Img2Img_Animation.webp" width="256"  alt="Image of browser inferencing on sample images."/> |

```
   Prompt: Dog wearing storm trooper helmet, head shot
```

### **Image Inpainting**
Image inpainting is an image modification/restoration technique that intelligently fills in missing or damaged portions of an image while maintaining visual consistency. It's used for tasks like photo restoration and object removal, creating seamless and convincing results.

In the below example we use a simple mask image + prompt to add a rider to the horse
The black part of the mask will be used buy the process to generate new content, in this case the rider

| Input Image | Mask Image | Masked Image | Result
| :--- | :--- | :--- | :--- |
<img src="../Assets/Samples/Inpaint-Original.png" width="256" alt="Image of browser inferencing on sample images."/> | <img src="../Assets/Samples/Inpaint-Mask.png" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="../Assets/Samples/Inpaint-MaskedImage.PNG" width="256"  alt="Image of browser inferencing on sample images."/> |<img src="../Assets/Samples/Inpaint-Result.png" width="256"  alt="Image of browser inferencing on sample images."/> |

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


## Resources
- [Hugging Face Stable Diffusion Blog](https://huggingface.co/blog/stable_diffusion)
- [ONNX Runtime tutorial for Stable Diffusion in C#](https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html)


## Reference
This work is based on the original C# implementation of Stable Diffusion by Cassie Breviu here: [Stable Diffusion with C# and ONNX Runtime](https://github.com/cassiebreviu/stablediffusion).
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

            System.Console.WriteLine("Please enter image filepath for Img2Img and press ENTER (optional)");
            var inputImageFile = System.Console.ReadLine();

            var promptOptions = new PromptOptions
            {
               Prompt = prompt,
               NegativePrompt = negativePrompt,
               SchedulerType = SchedulerType.LMSScheduler,
               InputImage = new InputImage
               {
                  ImagePath = inputImageFile
               }
            };

            var schedulerOptions = new SchedulerOptions
            {
               Seed = Random.Shared.Next(),
               GuidanceScale = 7.5f,
               InferenceSteps = 30,
               Height = 512,
               Width = 512,
               Strength = 0.6f // Img2Img
            };

            System.Console.WriteLine("Generating Image...");
            var outputFilename = Path.Combine(_outputDirectory, $"{schedulerOptions.Seed}_{promptOptions.SchedulerType}.png");
            var result = await _stableDiffusionService.GenerateAsImageAsync(prompt, options);
            if (result is not null)
            { 
               await result.SaveAsPngAsync(outputFilename);
               System.Console.WriteLine($"Image Created, FilePath: {outputFilename}");
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
	"AllowedHosts": "*",

	"OnnxStackConfig": {
		"Name": "StableDiffusion 1.5",
		"PadTokenId": 49407,
		"BlankTokenId": 49407,
		"InputTokenLimit": 512,
		"TokenizerLimit": 77,
		"EmbeddingsLength": 768,
		"ScaleFactor": 0.18215,
		"ModelConfigurations": [{
            "Type": "Unet",
            "DeviceId": 0,
            "ExecutionProvider": "DirectML",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\unet\\model.onnx"
         },
         {
            "Type": "Tokenizer",
            "DeviceId": 0,
            "ExecutionProvider": "Cpu",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\cliptokenizer.onnx"
         },
         {
            "Type": "TextEncoder",
            "DeviceId": 0,
            "ExecutionProvider": "Cpu",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\text_encoder\\model.onnx"
         },
         {
            "Type": "VaeEncoder",
            "DeviceId": 0,
            "ExecutionProvider": "Cpu",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\vae_encoder\\model.onnx"
         },
         {
            "Type": "VaeDecoder",
            "DeviceId": 0,
            "ExecutionProvider": "Cpu",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\vae_decoder\\model.onnx"
         },
         {
            "Type": "SafetyChecker",
            "IsDisabled": true,
            "DeviceId": 0,
            "ExecutionProvider": "Cpu",
            "OnnxModelPath": "D:\\Repositories\\stable-diffusion-v1-5\\safety_checker\\model.onnx"
         }]
	}
}
```
# OnnxStack.Core - Onnx Services for .NET Applications

OnnxStack.Core is a library that provides higher-level ONNX services for use in .NET applications. It offers extensive support for features such as dependency injection, .NET configuration implementations, ASP.NET Core integration, and IHostedService support.

You can configure a model set for runtime, offloading individual models to different devices to make better use of resources or run on lower-end hardware. The first use-case is StableDiffusion; however, it will be expanded, and other model sets, such as object detection and classification, will be added.

## Getting Started


OnnxStack.Core can be found via the nuget package manager, download and install it.
```
PM> Install-Package OnnxStack.Core
```


### .NET Core Registration

You can easily integrate `OnnxStack.Core` into your application services layer. This registration process sets up the necessary services and loads the `appsettings.json` configuration.

Example: Registering OnnxStack
```csharp
builder.Services.AddOnnxStack();
```

## Configuration example
The `appsettings.json` is the easiest option for configuring model sets. Below is an example of `Stable Diffusion 1.5`. The example adds the necessary paths to each model file required for Stable Diffusion, as well as any model-specific configurations. Each model can be assigned to its own device, which is handy if you have only a small GPU. This way, you can offload only what you need. There are limitations depending on the version of the `Microsoft.ML.OnnxRuntime` package you are using, but in most cases, you can split the load between CPU and GPU.

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
		"ModelConfigurations": [
			{
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
			}
		]
	}
}
```



### Basic C# Example
```csharp
// Create Configuration
var onnxStackConfig = new OnnxStackConfig
{
	Name = "OnnxStack",
	TokenizerLimit = 77,
	ModelConfigurations = new List<OnnxModelSessionConfig>
	{
		new OnnxModelSessionConfig
		{
				DeviceId = 0,
				ExecutionProvider = ExecutionProvider.DirectML,

				Type = OnnxModelType.Tokenizer,
				OnnxModelPath = "clip_tokenizer.onnx",
		}
	}
};

// Create Service
var onnxModelService = new OnnxModelService(onnxStackConfig);


// Tokenizer model Example
var text = "Text To Tokenize";
var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
var inputString = new List<NamedOnnxValue>
{
	NamedOnnxValue.CreateFromTensor("string_input", inputTensor)
};

// Create an InferenceSession from the Onnx clip tokenizer.
// Run session and send the input data in to get inference output. 
using (var tokens = onnxModelService.RunInference(OnnxModelType.Tokenizer, inputString))
{
	var resultTensor = tokens.ToArray();
}

```


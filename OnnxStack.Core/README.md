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

## Dependencies
Video processing support requires FFMPEG and FFPROBE binaries, files must be present in your output folder or the destinations configured in the `appsettings.json`
```
https://ffbinaries.com/downloads
https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffmpeg-6.1-win-64.zip
https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffprobe-6.1-win-64.zip
```

## Configuration example
The `appsettings.json` is the easiest option for configuring model sets. Below is an example of `clip tokenizer`.

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
		"OnnxModelSets": [
			{
				"Name": "ClipTokenizer",
				"IsEnabled": true,
				"DeviceId": 0,
				"InterOpNumThreads": 0,
				"IntraOpNumThreads": 0,
				"ExecutionMode": "ORT_SEQUENTIAL",
				"ExecutionProvider": "DirectML",
				"ModelConfigurations": [
					{
						"Type": "Tokenizer",
						"OnnxModelPath": "cliptokenizer.onnx"
					},
				]
			}
		]
	}
}
```



### Basic C# Example
```csharp

// Tokenizer model Example
//----------------------//

// From DI
OnnxStackConfig _onnxStackConfig;
IOnnxModelService _onnxModelService;

// Get Model
var model = _onnxStackConfig.OnnxModelSets.First();

// Get Model Metadata
var metadata = _onnxModelService.GetModelMetadata(model, OnnxModelType.Tokenizer);

// Create Input
var text = "Text To Tokenize";
var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });

// Create  Inference Parameters container
using (var inferenceParameters = new OnnxInferenceParameters(metadata))
{
	// Set Inputs and Outputs
	inferenceParameters.AddInputTensor(inputTensor);
	inferenceParameters.AddOutputBuffer();

	// Run Inference
	using (var results = _onnxModelService.RunInference(model, OnnxModelType.Tokenizer, inferenceParameters))
	{
		// Extract Result
		var resultData = results[0].ToDenseTensor();
	}
}

```


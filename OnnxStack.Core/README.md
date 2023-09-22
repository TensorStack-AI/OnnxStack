# OnnxStack.Core - Onnx Services for .NET Applications

OnnxStack.Core is a library that provides higher-level services for use in .NET applications. It offers extensive support for features such as dependency injection, .NET configuration implementations, ASP.NET Core integration, and IHostedService support.

## Getting Started

### .NET Core Registration

You can easily integrate `OnnxStack.Core` into your application services layer. This registration process sets up the necessary services and loads the `appsettings.json` configuration.

Example: Registering OnnxStack
```csharp
builder.Services.AddOnnxStack();
```




### Basic C# Example
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
    var resultTensor = tokens.FirstElementAs<Tensor<long>>();
}

```


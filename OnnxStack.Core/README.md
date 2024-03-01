# OnnxStack.Core - Onnx Services for .NET Applications

OnnxStack.Core is a library that provides simplified wrappers for OnnxRuntime

## Getting Started

OnnxStack.Core can be found via the nuget package manager, download and install it.
```
PM> Install-Package OnnxStack.Core
```

## Dependencies
Video processing support requires FFMPEG and FFPROBE binaries, files must be present in your output folder or the destinations configured at runtime
```
https://ffbinaries.com/downloads
https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffmpeg-6.1-win-64.zip
https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffprobe-6.1-win-64.zip
```


### OnnxModelSession Example
```csharp

// CLIP Tokenizer Example
//----------------------//

// Model Configuration
var config = new OnnxModelConfig
{
    DeviceId = 0,
    InterOpNumThreads = 0,
    IntraOpNumThreads = 0,
    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
    ExecutionProvider = ExecutionProvider.DirectML,
    OnnxModelPath = "cliptokenizer.onnx"
};

// Create Model Session
var modelSession = new OnnxModelSession(config);

// Get Metatdata
var modelMetadata = await modelSession.GetMetadataAsync();

// Create Input Tensor
var text = "Text To Tokenize";
var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });

// Create Inference Parameters
using (var inferenceParameters = new OnnxInferenceParameters(modelMetadata))
{
    // Set Inputs and Outputs
    inferenceParameters.AddInputTensor(inputTensor);
    inferenceParameters.AddOutputBuffer();

    // Run Inference
    using (var results = modelSession.RunInference(inferenceParameters))
    {
        // Extract Result Tokens
        var resultData = results[0].ToArray<long>();
    }
}

```


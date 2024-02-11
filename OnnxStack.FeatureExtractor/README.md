# OnnxStack.FeatureExtractor

## Canny
https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/canny.onnx

## Hed
https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/hed.onnx

## Depth
https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/depth.onnx
https://huggingface.co/Xenova/depth-anything-large-hf/onnx/model.onnx
https://huggingface.co/julienkay/sentis-MiDaS

## OpenPose (TODO)
https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/openpose.onnx

# Basic Example
```csharp

// Load Input Image
var inputImage = await InputImage.FromFileAsync("Input.png");

// Load Pipeline
var pipeline = FeatureExtractorPipeline.CreatePipeline("canny.onnx");

// Run Pipeline
var imageFeature = await pipeline.RunAsync(inputImage);

// Save Image
await imageFeature.Image.SaveAsPngAsync("Result.png");

//Unload
await pipeline.UnloadAsync();
 ```
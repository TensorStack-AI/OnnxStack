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

# Image Example
```csharp

// Load Input Image
var inputImage = await OnnxImage.FromFileAsync("Input.png");

// Load Pipeline
var pipeline = FeatureExtractorPipeline.CreatePipeline("canny.onnx");

// Run Pipeline
var imageFeature = await pipeline.RunAsync(inputImage);

// Save Image
await imageFeature.Image.SaveAsync("Result.png");

//Unload
await pipeline.UnloadAsync();
 ```

 # Video Example
```csharp

// Load Input Video
var inputVideo = await OnnxVideo.FromFileAsync("Input.mp4");

// Load Pipeline
var pipeline = FeatureExtractorPipeline.CreatePipeline("canny.onnx");

// Run Pipeline
var videoFeature = await pipeline.RunAsync(inputVideo);

// Save Video
await videoFeature.SaveAsync("Result.mp4");

//Unload
await pipeline.UnloadAsync();
 ```
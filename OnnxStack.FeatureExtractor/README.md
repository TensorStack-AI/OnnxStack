# OnnxStack.FeatureExtractor

### Canny
* https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/canny.onnx

### Hed
* https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/hed.onnx

### Depth
* https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/depth.onnx
* https://huggingface.co/Xenova/depth-anything-large-hf/onnx/model.onnx
* https://huggingface.co/julienkay/sentis-MiDaS

### OpenPose (TODO)
* https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/openpose.onnx

### Background Removal
* https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx


# Image Example
```csharp
// Load Input Image
var inputImage = await OnnxImage.FromFileAsync("Input.png");

// Load Pipeline
var pipeline = FeatureExtractorPipeline.CreatePipeline("canny.onnx");

// Run Pipeline
var result = await pipeline.RunAsync(inputImage);

// Save Image
await result.SaveAsync("Result.png");

//Unload
await pipeline.UnloadAsync();
 ```

 # Video Example
```csharp
// Load Input Image
var inputImage = await OnnxVideo.FromFileAsync("Input.mp4");

// Load Pipeline
var pipeline = FeatureExtractorPipeline.CreatePipeline("canny.onnx");

// Run Pipeline
var result = await pipeline.RunAsync(inputImage);

// Save Image
await result.SaveAsync("Result.mp4");

//Unload
await pipeline.UnloadAsync();
 ```

  # Video Stream Example
```csharp
// Read Video Info
var videoFile = "Input.mp4";
var videoInfo = await VideoHelper.ReadVideoInfoAsync(videoFile);

// Create Video Stream
var videoStream = VideoHelper.ReadVideoStreamAsync(videoFile, videoInfo.FrameRate);

// Create pipeline
var pipeline = FeatureExtractorPipeline.CreatePipeline("canny.onnx");

// Create Pipeline Stream
var pipelineStream = pipeline.RunAsync(videoStream);

// Write Video Stream
await VideoHelper.WriteVideoStreamAsync(videoInfo, pipelineStream, "Result.mp4");

//Unload
await pipeline.UnloadAsync();
 ```
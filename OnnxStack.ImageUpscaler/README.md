# OnnxStack.ImageUpscaler

## Upscale Models
Below is a small list of known/tested upscale models
* https://huggingface.co/wuminghao/swinir
* https://huggingface.co/rocca/swin-ir-onnx
* https://huggingface.co/Xenova/swin2SR-classical-sr-x2-64
* https://huggingface.co/Xenova/swin2SR-classical-sr-x4-64
* https://huggingface.co/Neus/GFPGANv1.4


# Image Example
```csharp
// Load Input Image
var inputImage = await OnnxImage.FromFileAsync("Input.png");

// Create Pipeline
var pipeline = ImageUpscalePipeline.CreatePipeline("SwinIR-M_x4_GAN.onnx", scaleFactor: 4);

// Run pipeline
var result = await pipeline.RunAsync(inputImage);

// Save Image File
await result.SaveAsync("Result.png");

// Unload
await pipeline.UnloadAsync();
```


# Video Example
```csharp
// Load Input Video
var inputVideo = await OnnxVideo.FromFileAsync("Input.mp4");

// Create Pipeline
var pipeline = ImageUpscalePipeline.CreatePipeline("SwinIR-M_x4_GAN.onnx", scaleFactor: 4);

// Run pipeline
var result = await pipeline.RunAsync(inputVideo);

// Save Video File
await result.SaveAsync("Result.mp4");

// Unload
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
var pipeline = ImageUpscalePipeline.CreatePipeline("SwinIR-M_x4_GAN.onnx", scaleFactor: 4);

// Create Pipeline Stream
var pipelineStream = pipeline.RunAsync(videoStream);

// Write Video Stream
await VideoHelper.WriteVideoStreamAsync(videoInfo, pipelineStream, "Result.mp4");

//Unload
await pipeline.UnloadAsync();
```
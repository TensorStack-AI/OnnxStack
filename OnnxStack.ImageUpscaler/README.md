# OnnxStack.ImageUpscaler

## Upscale Models
https://huggingface.co/wuminghao/swinir
https://huggingface.co/rocca/swin-ir-onnx
https://huggingface.co/Xenova/swin2SR-classical-sr-x2-64
https://huggingface.co/Xenova/swin2SR-classical-sr-x4-64


# Basic Example
```csharp
// Load Input Image
var inputImage = await OnnxImage.FromFileAsync("Input.png");

// Create Pipeline
var pipeline = ImageUpscalePipeline.CreatePipeline("003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx", scaleFactor: 4);

// Run pipeline
var result = await pipeline.RunAsync(inputImage);

// Create Image from Tensor result
var image = new OnnxImage(result, ImageNormalizeType.ZeroToOne);

// Save Image File
await image.SaveAsync("Upscaled.png");

// Unload
await pipeline.UnloadAsync();
```
using OnnxStack.Core.Image;
using OnnxStack.FeatureExtractor.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class UpscaleExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public UpscaleExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(UpscaleExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 10;

        public string Name => "Image Upscale Demo";

        public string Description => "Upscales images";

        public async Task RunAsync()
        {
            // Load Input Image
            var inputImage = await OnnxImage.FromFileAsync("D:\\Repositories\\OnnxStack\\Assets\\Samples\\Img2Img_Start.bmp");

            // Create Pipeline
            var pipeline = ImageUpscalePipeline.CreatePipeline("D:\\Repositories\\upscaler\\SwinIR\\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx", 4, 512);

            // Run pipeline
            var result = await pipeline.RunAsync(inputImage);
         
            // Save Image File
            var outputFilename = Path.Combine(_outputDirectory, $"Upscaled.png");
            await result.SaveAsync(outputFilename);

            // Unload
            await pipeline.UnloadAsync();
        }

    }
}

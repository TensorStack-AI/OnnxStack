using OnnxStack.Core.Image;
using OnnxStack.FeatureExtractor.Common;
using OnnxStack.FeatureExtractor.Pipelines;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class ControlNetFeatureExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public ControlNetFeatureExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(ControlNetFeatureExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 12;

        public string Name => "ControlNet + Feature Extraction Example";

        public string Description => "ControlNet StableDiffusion with input image Depth feature extraction";

        /// <summary>
        /// ControlNet Example
        /// </summary>
        public async Task RunAsync()
        {
            // Execution provider
            var provider = Providers.DirectML(0);

            // Load Control Image
            var inputImage = await OnnxImage.FromFileAsync("D:\\Repositories\\OnnxStack\\Assets\\Samples\\Img2Img_Start.bmp");

            // Create FeatureExtractor
            var featureExtractor = FeatureExtractorPipeline.CreatePipeline(provider, "M:\\Models\\FeatureExtractor-onnx\\SD\\Depth\\model.onnx", sampleSize: 512, normalizeOutputType: ImageNormalizeType.MinMax);

            // Create Depth Image
            var controlImage = await featureExtractor.RunAsync(inputImage, new FeatureExtractorOptions());

            // Save Depth Image (Debug Only)
            await controlImage.SaveAsync(Path.Combine(_outputDirectory, $"Depth.png"));

            // Create ControlNet
            var controlNet = ControlNetModel.Create(provider, "M:\\Models\\ControlNet-onnx\\SD\\Depth\\model.onnx");

            // Create Pipeline
            var pipeline = StableDiffusionPipeline.CreatePipeline(provider, "M:\\Models\\stable-diffusion-v1-5-onnx");


            var generateOptions = new GenerateOptions
            {
                Prompt = "cyberpunk dog",
                Diffuser = DiffuserType.ControlNet,
                InputContolImage = controlImage,
                ControlNet = controlNet,
                SchedulerOptions = pipeline.DefaultSchedulerOptions with
                {
                    Seed = 12345,
                    ConditioningScale = 0.8f
                }
            };

            // Run pipeline
            var result = await pipeline.RunAsync(generateOptions, progressCallback: OutputHelpers.ProgressCallback);

            // Create Image from Tensor result
            var image = new OnnxImage(result);

            // Save Image File
            var outputFilename = Path.Combine(_outputDirectory, $"Output.png");
            await image.SaveAsync(outputFilename);

            //Unload
            await featureExtractor.UnloadAsync();
            await controlNet.UnloadAsync();
            await pipeline.UnloadAsync();
        }
    }
}

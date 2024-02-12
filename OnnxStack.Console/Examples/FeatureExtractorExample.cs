using OnnxStack.Core.Image;
using OnnxStack.FeatureExtractor.Pipelines;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class FeatureExtractorExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public FeatureExtractorExample(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(FeatureExtractorExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 13;

        public string Name => "Feature Extractor Example";

        public string Description => "Simple exmaple using ControlNet feature extractors";

        /// <summary>
        /// ControlNet Example
        /// </summary>
        public async Task RunAsync()
        {
            // Load Control Image
            var inputImage = await OnnxImage.FromFileAsync("D:\\Repositories\\OnnxStack\\Assets\\Samples\\Img2Img_Start.bmp");

            var pipelines = new[]
            {
                FeatureExtractorPipeline.CreatePipeline("D:\\Repositories\\controlnet_onnx\\annotators\\canny.onnx"),
                FeatureExtractorPipeline.CreatePipeline("D:\\Repositories\\controlnet_onnx\\annotators\\hed.onnx"),
                FeatureExtractorPipeline.CreatePipeline("D:\\Repositories\\controlnet_onnx\\annotators\\depth.onnx", true),

               // FeatureExtractorPipeline.CreatePipeline("D:\\Repositories\\depth-anything-large-hf\\onnx\\model.onnx", normalize: true, sampleSize: 504),
               // FeatureExtractorPipeline.CreatePipeline("D:\\Repositories\\sentis-MiDaS\\dpt_beit_large_512.onnx", normalize: true, sampleSize: 384),
            };

            foreach (var pipeline in pipelines)
            {
                var timestamp = Stopwatch.GetTimestamp();
                OutputHelpers.WriteConsole($"Load pipeline`{pipeline.Name}`", ConsoleColor.Cyan);

                // Run Pipeline
                var imageFeature = await pipeline.RunAsync(inputImage);

                OutputHelpers.WriteConsole($"Generating image", ConsoleColor.Cyan);

                // Save Image
                await imageFeature.SaveAsync(Path.Combine(_outputDirectory, $"{pipeline.Name}.png"));


                OutputHelpers.WriteConsole($"Unload pipeline", ConsoleColor.Cyan);

                //Unload
                await pipeline.UnloadAsync();

                OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);
            }
        }
    }
}

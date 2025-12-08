using OnnxStack.Core.Image;
using OnnxStack.FeatureExtractor.Common;
using OnnxStack.FeatureExtractor.Pipelines;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class FeatureExtractorExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public FeatureExtractorExample()
        {
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
            // Execution provider
            var provider = Providers.DirectML(0);

            // Load Control Image
            var inputImage = await OnnxImage.FromFileAsync("D:\\Repositories\\OnnxStack\\Assets\\Samples\\Img2Img_Start.bmp");

            var pipelines = new[]
            {
                FeatureExtractorPipeline.CreatePipeline(provider, "D:\\Repositories\\controlnet_onnx\\annotators\\canny.onnx"),
                FeatureExtractorPipeline.CreatePipeline(provider, "D:\\Repositories\\controlnet_onnx\\annotators\\hed.onnx"),
                FeatureExtractorPipeline.CreatePipeline(provider, "D:\\Repositories\\controlnet_onnx\\annotators\\depth.onnx", sampleSize: 512, normalizeOutputType: ImageNormalizeType.MinMax, inputResizeMode: ImageResizeMode.Stretch),
                FeatureExtractorPipeline.CreatePipeline(provider, "D:\\Repositories\\RMBG-1.4\\onnx\\model.onnx", sampleSize: 1024, setOutputToInputAlpha: true, inputResizeMode: ImageResizeMode.Stretch)
            };

            foreach (var pipeline in pipelines)
            {
                var timestamp = Stopwatch.GetTimestamp();
                OutputHelpers.WriteConsole($"Load pipeline`{pipeline.Name}`", ConsoleColor.Cyan);

                // Run Image Pipeline
                var imageFeature = await pipeline.RunAsync(inputImage.Clone(), new FeatureExtractorOptions());

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

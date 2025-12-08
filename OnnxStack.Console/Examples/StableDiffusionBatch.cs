using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionBatch : IExampleRunner
    {
        private readonly string _outputDirectory;

        public StableDiffusionBatch()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionBatch));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 3;

        public string Name => "Stable Diffusion Batch Demo";

        public string Description => "Creates a batch of images from the text prompt provided using all Scheduler types";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            // Execution provider
            var provider = Providers.DirectML(0);

            // Create Pipeline
            var pipeline = StableDiffusionPipeline.CreatePipeline(provider, "M:\\Models\\stable-diffusion-v1-5-onnx");
            OutputHelpers.WriteConsole($"Loading Model `{pipeline.Name}`...", ConsoleColor.Green);

            // Run Batch
            var timestamp = Stopwatch.GetTimestamp();

            // Options
            var generateOptions = new GenerateBatchOptions
            {
                Prompt = "Photo of a cat",

                // Batch Of 5 seeds
                ValueTo = 5,
                BatchType = BatchOptionType.Seed
            };

            // Generate
            await foreach (var result in pipeline.RunBatchAsync(generateOptions, progressCallback: OutputHelpers.BatchProgressCallback))
            {
                // Create Image from Tensor result
                var image = new OnnxImage(result.Result);

                // Save Image File
                var outputFilename = Path.Combine(_outputDirectory, $"{pipeline.Name}_{result.SchedulerOptions.Seed}.png");
                await image.SaveAsync(outputFilename);

                OutputHelpers.WriteConsole($"Image Created: {Path.GetFileName(outputFilename)}, Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Green);
                timestamp = Stopwatch.GetTimestamp();
            }

            OutputHelpers.WriteConsole($"Unloading Model `{pipeline.Name}`...", ConsoleColor.Green);

            // Unload
            await pipeline.UnloadAsync();
        }
    }
}

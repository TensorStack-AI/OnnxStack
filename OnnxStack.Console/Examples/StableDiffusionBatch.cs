using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;
using SixLabors.ImageSharp;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionBatch : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public StableDiffusionBatch(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionBatch));
        }

        public int Index => 2;

        public string Name => "Stable Diffusion Batch Demo";

        public string Description => "Creates a batch of images from the text prompt provided using all Scheduler types";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);


            // Prompt
            var promptOptions = new PromptOptions
            {
                Prompt = "Photo of a cat"
            };

            // Batch Of 5
            var batchOptions = new BatchOptions
            {
                ValueTo = 5,
                BatchType = BatchOptionType.Seed
            };

            foreach (var modelSet in _configuration.ModelSets)
            {
                OutputHelpers.WriteConsole($"Loading Model `{modelSet.Name}`...", ConsoleColor.Green);

                // Create Pipeline
                var pipeline = PipelineBase.CreatePipeline(modelSet);

                // Preload Models (optional)
                await pipeline.LoadAsync();

                // Run Batch
                var timestamp = Stopwatch.GetTimestamp();
                await foreach (var result in pipeline.RunBatchAsync(batchOptions, promptOptions, progressCallback: OutputHelpers.BatchProgressCallback))
                {
                    // Create Image from Tensor result
                    var image = result.ImageResult;

                    // Save Image File
                    var outputFilename = Path.Combine(_outputDirectory, $"{modelSet.Name}_{result.SchedulerOptions.Seed}.png");
                    await image.SaveAsync(outputFilename);

                    OutputHelpers.WriteConsole($"Image Created: {Path.GetFileName(outputFilename)}, Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Green);
                    timestamp = Stopwatch.GetTimestamp();
                }

                OutputHelpers.WriteConsole($"Unloading Model `{modelSet.Name}`...", ConsoleColor.Green);

                // Unload
                await pipeline.UnloadAsync();

            }
        }
    }
}

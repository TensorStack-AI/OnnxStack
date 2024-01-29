using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Pipelines;
using SixLabors.ImageSharp;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public StableDiffusionExample(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionExample));
        }

        public int Index => 1;

        public string Name => "Stable Diffusion Demo";

        public string Description => "Creates images from the text prompt provided using all Scheduler types";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);

            while (true)
            {
                OutputHelpers.WriteConsole("Please type a prompt and press ENTER", ConsoleColor.Yellow);
                var prompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                OutputHelpers.WriteConsole("Please type a negative prompt and press ENTER (optional)", ConsoleColor.Yellow);
                var negativePrompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                var promptOptions = new PromptOptions
                {
                    Prompt = prompt,
                    NegativePrompt = negativePrompt,
                };

                foreach (var modelSet in _configuration.ModelSets)
                {
                    OutputHelpers.WriteConsole($"Loading Model `{modelSet.Name}`...", ConsoleColor.Green);

                    // Create Pipeline
                    var pipeline = PipelineBase.CreatePipeline(modelSet);

                    // Preload Models (optional)
                    await pipeline.LoadAsync();


                    // Loop through schedulers
                    foreach (var schedulerType in pipeline.SupportedSchedulers)
                    {
                        var schedulerOptions = pipeline.DefaultSchedulerOptions with
                        {
                            SchedulerType = schedulerType
                        };

                        var timestamp = Stopwatch.GetTimestamp();
                        OutputHelpers.WriteConsole($"Generating '{schedulerType}' Image...", ConsoleColor.Green);

                        // Run pipeline
                        var result = await pipeline.RunAsync(promptOptions, schedulerOptions);

                        // Create Image from Tensor result
                        var image = result.ToImage();

                        // Save Image File
                        var outputFilename = Path.Combine(_outputDirectory, $"{modelSet.Name}_{schedulerOptions.SchedulerType}.png");
                        await image.SaveAsPngAsync(outputFilename);

                        OutputHelpers.WriteConsole($"Image Created: {Path.GetFileName(outputFilename)}, Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Green);
                    }

                    OutputHelpers.WriteConsole($"Unloading Model `{modelSet.Name}`...", ConsoleColor.Green);
                    await pipeline.UnloadAsync();
                }
            }
        }
    }
}

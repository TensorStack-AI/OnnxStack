using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Pipelines;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public StableDiffusionExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 3;

        public string Name => "Stable Diffusion Demo";

        public string Description => "Creates images from the text prompt provided using all Scheduler types";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            // Execution provider
            var provider = Providers.DirectML(0);

            while (true)
            {
                OutputHelpers.WriteConsole("Please type a prompt and press ENTER", ConsoleColor.Yellow);
                var prompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                OutputHelpers.WriteConsole("Please type a negative prompt and press ENTER (optional)", ConsoleColor.Yellow);
                var negativePrompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);


                // Create Pipeline
                var pipeline = StableDiffusionPipeline.CreatePipeline(provider, "M:\\Models\\stable-diffusion-v1-5-onnx");
                OutputHelpers.WriteConsole($"Loading Model `{pipeline.Name}`...", ConsoleColor.Green);

                // Options
                var generateOptions = new GenerateOptions
                {
                    Prompt = prompt,
                    NegativePrompt = negativePrompt
                };

                // Loop through schedulers
                foreach (var schedulerType in pipeline.SupportedSchedulers)
                {
                    generateOptions.SchedulerOptions = pipeline.DefaultSchedulerOptions with
                    {
                        SchedulerType = schedulerType
                    };

                    var timestamp = Stopwatch.GetTimestamp();
                    OutputHelpers.WriteConsole($"Generating '{schedulerType}' Image...", ConsoleColor.Green);

                    // Run pipeline
                    var result = await pipeline.GenerateAsync(generateOptions, progressCallback: OutputHelpers.ProgressCallback);

                    // Save Image File
                    var outputFilename = Path.Combine(_outputDirectory, $"{pipeline.Name}_{generateOptions.SchedulerOptions.SchedulerType}.png");
                    await result.SaveAsync(outputFilename);

                    OutputHelpers.WriteConsole($"Image Created: {Path.GetFileName(outputFilename)}, Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Green);
                }

                OutputHelpers.WriteConsole($"Unloading Model `{pipeline.Name}`...", ConsoleColor.Green);
                await pipeline.UnloadAsync();
            }

        }
    }
}

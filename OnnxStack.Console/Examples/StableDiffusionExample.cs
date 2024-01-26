using OnnxStack.StableDiffusion;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;
        private readonly IStableDiffusionService _stableDiffusionService;

        public StableDiffusionExample(StableDiffusionConfig configuration, IStableDiffusionService stableDiffusionService)
        {
            _configuration = configuration;
            _stableDiffusionService = stableDiffusionService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionExample));
        }

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

                var schedulerOptions = new SchedulerOptions
                {
                    Seed = Random.Shared.Next()
                };

                foreach (var model in _configuration.ModelSets)
                {
                    OutputHelpers.WriteConsole($"Loading Model `{model.Name}`...", ConsoleColor.Green);
                    await _stableDiffusionService.LoadModelAsync(model);

                    schedulerOptions.Width = model.SampleSize;
                    schedulerOptions.Height = model.SampleSize;
                    
                    foreach (var schedulerType in model.PipelineType.GetSchedulerTypes())
                    {
                        schedulerOptions.SchedulerType = schedulerType;
                        OutputHelpers.WriteConsole($"Generating {schedulerType} Image...", ConsoleColor.Green);
                        await GenerateImage(model, promptOptions, schedulerOptions);
                    }

                    OutputHelpers.WriteConsole($"Unloading Model `{model.Name}`...", ConsoleColor.Green);
                    await _stableDiffusionService.UnloadModelAsync(model);
                }
            }
        }

        private async Task<bool> GenerateImage(StableDiffusionModelSet model, PromptOptions prompt, SchedulerOptions options)
        {
            var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{options.SchedulerType}.png");
            var result = await _stableDiffusionService.GenerateAsImageAsync(new ModelOptions(model), prompt, options);
            if (result == null)
                return false;

            await result.SaveAsPngAsync(outputFilename);
            OutputHelpers.WriteConsole($"{options.SchedulerType} Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
            return true;
        }
    }
}

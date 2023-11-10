using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionBatch : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly IStableDiffusionService _stableDiffusionService;

        public StableDiffusionBatch(IStableDiffusionService stableDiffusionService)
        {
            _stableDiffusionService = stableDiffusionService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionBatch));
        }

        public string Name => "Stable Diffusion Batch Demo";

        public string Description => "Creates a batch of images from the text prompt provided using all Scheduler types";

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

                OutputHelpers.WriteConsole("Please enter a batch count and press ENTER", ConsoleColor.Yellow);
                var batch = OutputHelpers.ReadConsole(ConsoleColor.Cyan);
                int.TryParse(batch, out var batchCount);
                batchCount = Math.Max(1, batchCount);

                var promptOptions = new PromptOptions
                {
                    Prompt = prompt,
                    NegativePrompt = negativePrompt,
                    BatchCount = batchCount
                };

                var schedulerOptions = new SchedulerOptions
                {
                    Seed = Random.Shared.Next(),

                    GuidanceScale = 8,
                    InferenceSteps = 22,
                    Strength = 0.6f
                };

                foreach (var model in _stableDiffusionService.Models)
                {
                    OutputHelpers.WriteConsole($"Loading Model `{model.Name}`...", ConsoleColor.Green);
                    await _stableDiffusionService.LoadModel(model);

                    foreach (var schedulerType in Helpers.GetPipelineSchedulers(model.PipelineType))
                    {
                        promptOptions.SchedulerType = schedulerType;
                        OutputHelpers.WriteConsole($"Generating {schedulerType} Image...", ConsoleColor.Green);
                        await GenerateImage(model, promptOptions, schedulerOptions);
                    }

                    OutputHelpers.WriteConsole($"Unloading Model `{model.Name}`...", ConsoleColor.Green);
                    await _stableDiffusionService.UnloadModel(model);
                }
            }
        }

        private async Task<bool> GenerateImage(ModelOptions model, PromptOptions prompt, SchedulerOptions options)
        {

            var result = await _stableDiffusionService.GenerateAsync(model, prompt, options);
            if (result == null)
                return false;

            var imageTensors = result.Split(prompt.BatchCount);
            for (int i = 0; i < imageTensors.Length; i++)
            {
                var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{prompt.SchedulerType}_{i}.png");
                var image = imageTensors[i].ToImage();
                await image.SaveAsPngAsync(outputFilename);
                OutputHelpers.WriteConsole($"{prompt.SchedulerType} Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
            }

            return true;
        }
    }
}

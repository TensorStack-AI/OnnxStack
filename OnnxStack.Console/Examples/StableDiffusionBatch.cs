using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using SixLabors.ImageSharp;

namespace OnnxStack.Console.Runner
{
    using StableDiffusion;

    public sealed class StableDiffusionBatch : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;
        private readonly IStableDiffusionService _stableDiffusionService;

        public StableDiffusionBatch(StableDiffusionConfig configuration, IStableDiffusionService stableDiffusionService)
        {
            _configuration = configuration;
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

                var promptOptions = new PromptOptions
                {
                    Prompt = "Photo of a cat"
                };

                var schedulerOptions = new SchedulerOptions
                {
                    Seed = Random.Shared.Next(),

                    GuidanceScale = 8,
                    InferenceSteps = 20,
                    Strength = 0.6f
                };

                var batchOptions = new BatchOptions
                {
                    BatchType = BatchOptionType.Scheduler
                };

                foreach (var model in _configuration.ModelSets)
                {
                    _setSchedulerTypeForPipeline();

                    OutputHelpers.WriteConsole($"Loading Model `{model.Name}`...", ConsoleColor.Green);
                    await _stableDiffusionService.LoadModelAsync(model);

                    var batchIndex = 0;
                    var callback = (DiffusionProgress progress) =>
                    {
                        batchIndex = progress.BatchValue;
                        OutputHelpers.WriteConsole($"Image: {progress.BatchValue}/{progress.BatchMax} - Step: {progress.StepValue}/{progress.StepMax}", ConsoleColor.Cyan);
                    };

                    await foreach (var result in _stableDiffusionService.GenerateBatchAsync(new ModelOptions(model), promptOptions, schedulerOptions, batchOptions, default))
                    {
                        var outputFilename = Path.Combine(_outputDirectory, $"{batchIndex}_{result.SchedulerOptions.Seed}.png");
                        var image = result.ImageResult.ToImage();
                        await image.SaveAsPngAsync(outputFilename);
                        OutputHelpers.WriteConsole($"Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
                    }

                    OutputHelpers.WriteConsole($"Unloading Model `{model.Name}`...", ConsoleColor.Green);
                    await _stableDiffusionService.UnloadModelAsync(model);
                    continue;

                    void _setSchedulerTypeForPipeline()
                    {
                        SchedulerType[] scheduleTypes = model.PipelineType.GetSchedulerTypes();
                        schedulerOptions.SchedulerType = scheduleTypes.Length == 1 ? scheduleTypes[0] : scheduleTypes[Random.Shared.Next(scheduleTypes.Length)];
                    }
                }
            }
        }
    }
}

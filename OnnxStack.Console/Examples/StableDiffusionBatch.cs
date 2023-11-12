using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
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
                    BatchType = BatchOptionType.Guidance,
                    ValueFrom = 4,
                    ValueTo = 20,
                    Increment = 0.5f
                };

                foreach (var model in _stableDiffusionService.Models)
                {
                    OutputHelpers.WriteConsole($"Loading Model `{model.Name}`...", ConsoleColor.Green);
                    await _stableDiffusionService.LoadModel(model);

                    var batchIndex = 0;
                    var callback = (int batch, int batchCount, int step, int steps) =>
                    {
                        batchIndex = batch;
                        OutputHelpers.WriteConsole($"Image: {batch}/{batchCount} - Step: {step}/{steps}", ConsoleColor.Cyan);
                    };

                    foreach (var schedulerType in Helpers.GetPipelineSchedulers(model.PipelineType).Take(1))
                    {
                        promptOptions.SchedulerType = schedulerType;
                        await foreach (var result in _stableDiffusionService.GenerateBatchAsync(model, promptOptions, schedulerOptions, batchOptions, callback))
                        {
                            var outputFilename = Path.Combine(_outputDirectory, $"{batchIndex}_{result.SchedulerOptions.Seed}.png");
                            var image = result.ImageResult.ToImage();
                            await image.SaveAsPngAsync(outputFilename);
                            OutputHelpers.WriteConsole($"Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
                        }
                    }

                    OutputHelpers.WriteConsole($"Unloading Model `{model.Name}`...", ConsoleColor.Green);
                    await _stableDiffusionService.UnloadModel(model);
                }
            }
        }
    }
}

using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Pipelines;
using SixLabors.ImageSharp;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class StableCascadeExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public StableCascadeExample(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableCascadeExample));
        }

        public int Index => 20;

        public string Name => "Stable Cascade Demo";

        public string Description => "Creates images from the text prompt provided";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);


            var prompt = "photo of a cat";
            var promptOptions = new PromptOptions
            {
                Prompt = prompt
            };

            // Create Pipeline
            var pipeline = StableCascadePipeline.CreatePipeline("D:\\Repositories\\stable-cascade-onnx", memoryMode: StableDiffusion.Enums.MemoryModeType.Minimum);

            // Preload Models (optional)
            await pipeline.LoadAsync();


            // Loop through schedulers
            var schedulerOptions = pipeline.DefaultSchedulerOptions with
            {
                SchedulerType = StableDiffusion.Enums.SchedulerType.DDPM,
                GuidanceScale =4f,
                InferenceSteps = 60,
                Width = 1024,
                Height = 1024
            };

            var timestamp = Stopwatch.GetTimestamp();


            // Run pipeline
            var result = await pipeline.RunAsync(promptOptions, schedulerOptions, progressCallback: OutputHelpers.ProgressCallback);

            var image = new OnnxImage(result, ImageNormalizeType.ZeroToOne);

            // Save Image File
            await image.SaveAsync(Path.Combine(_outputDirectory, $"output.png"));

            await pipeline.UnloadAsync();


        }
    }
}

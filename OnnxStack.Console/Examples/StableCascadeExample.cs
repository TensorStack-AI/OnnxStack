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
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionExample));
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


            var prompt = "cat wearing a hat";
            var promptOptions = new PromptOptions
            {
                Prompt = prompt
            };

            // Create Pipeline
            var pipeline = StableCascadePipeline.CreatePipeline("D:\\Repositories\\stable-cascade-onnx\\unoptimized", memoryMode: StableDiffusion.Enums.MemoryModeType.Minimum);

            // Preload Models (optional)
            await pipeline.LoadAsync();


            // Loop through schedulers
            var schedulerOptions = pipeline.DefaultSchedulerOptions with
            {
                SchedulerType = StableDiffusion.Enums.SchedulerType.DDPM,
                GuidanceScale = 5f,
                InferenceSteps = 10,
                Width = 1024,
                Height = 1024
            };

            var timestamp = Stopwatch.GetTimestamp();


            // Run pipeline
            var result = await pipeline.GenerateImageAsync(promptOptions, schedulerOptions, progressCallback: OutputHelpers.ProgressCallback);

            // Save Image File
            await result.SaveAsync(Path.Combine(_outputDirectory, $"output.png"));

            await pipeline.UnloadAsync();


        }
    }
}

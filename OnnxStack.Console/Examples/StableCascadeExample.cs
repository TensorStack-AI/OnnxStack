using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
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
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 20;

        public string Name => "Stable Cascade Demo";

        public string Description => "Creates images from the text prompt provided";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            // Prompt
            var promptOptions = new PromptOptions
            {
                Prompt = "an image of a cat, donning a spacesuit and helmet",
                DiffuserType = DiffuserType.TextToImage,
                //InputImage = await OnnxImage.FromFileAsync("Input.png"),
            };

            // Create Pipeline
            var pipeline = StableCascadePipeline.CreatePipeline("D:\\Models\\stable-cascade-onnx", memoryMode: MemoryModeType.Minimum);

            // Run pipeline
            var result = await pipeline.GenerateImageAsync(promptOptions, progressCallback: OutputHelpers.ProgressCallback);

            // Save Image File
            await result.SaveAsync(Path.Combine(_outputDirectory, $"output.png"));

            // Unload
            await pipeline.UnloadAsync();
        }
    }
}

using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class StableCascadeExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public StableCascadeExample()
        {
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
            // Execution provider
            var provider = Providers.DirectML(0);

            // Prompt
            var generateOptions = new GenerateOptions
            {
                Prompt = "an image of a cat, donning a spacesuit and helmet",
                Diffuser = DiffuserType.TextToImage,
            };

            // Create Pipeline
            var pipeline = StableCascadePipeline.CreatePipeline(provider, "D:\\Models\\stable-cascade-onnx");

            // Run pipeline
            var result = await pipeline.GenerateAsync(generateOptions, progressCallback: OutputHelpers.ProgressCallback);

            // Save Image File
            await result.SaveAsync(Path.Combine(_outputDirectory, $"output.png"));

            // Unload
            await pipeline.UnloadAsync();
        }
    }
}

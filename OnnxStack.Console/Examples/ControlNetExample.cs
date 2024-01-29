using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Pipelines;
using SixLabors.ImageSharp;

namespace OnnxStack.Console.Runner
{
    public sealed class ControlNetExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public ControlNetExample(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(ControlNetExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 11;

        public string Name => "ControlNet Example";

        public string Description => "ControlNet Example";

        /// <summary>
        /// ControlNet Example
        /// </summary>
        public async Task RunAsync()
        {
            // Load Control Image
            var controlImage = await File.ReadAllBytesAsync("D:\\Repositories\\OnnxStack\\Assets\\Samples\\OpenPose.png");

            // Create ControlNet
            var controlNet = ControlNetModel.Create("D:\\Repositories\\controlnet_onnx\\controlnet\\openpose.onnx");

            // Create Pipeline
            var pipeline = StableDiffusionPipeline.CreatePipeline("D:\\Repositories\\stable_diffusion_onnx", ModelType.ControlNet);

            // Prompt
            var promptOptions = new PromptOptions
            {
                Prompt = "Stormtrooper flexing",
                DiffuserType = DiffuserType.ControlNet,
                InputContolImage = new InputImage(controlImage)
            };

            // Run pipeline
            var result = await pipeline.RunAsync(promptOptions, controlNet: controlNet);

            // Create Image from Tensor result
            var image = result.ToImage();

            // Save Image File
            var outputFilename = Path.Combine(_outputDirectory, $"Output.png");
            await image.SaveAsPngAsync(outputFilename);

        }
    }
}

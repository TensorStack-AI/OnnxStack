using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Common;
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
            var controlImage = await OnnxImage.FromFileAsync("D:\\Repositories\\OnnxStack\\Assets\\Samples\\OpenPose.png");

            // Create ControlNet
            var controlNet = ControlNetModel.Create("D:\\Models\\controlnet_onnx\\controlnet\\openpose.onnx", ControlNetType.OpenPose, DiffuserPipelineType.StableDiffusion);

            // Create Pipeline
            var pipeline = StableDiffusionPipeline.CreatePipeline("D:\\Models\\stable-diffusion-v1-5-onnx");

            // Prompt
            var promptOptions = new PromptOptions
            {
                Prompt = "Stormtrooper",
                DiffuserType = DiffuserType.ControlNet,
                InputContolImage = controlImage
            };

            // Preload (optional)
            await pipeline.LoadAsync(true);

            // Run pipeline
            var result = await pipeline.RunAsync(promptOptions, controlNet: controlNet, progressCallback: OutputHelpers.ProgressCallback);

            // Create Image from Tensor result
            var image = new OnnxImage(result);

            // Save Image File
            var outputFilename = Path.Combine(_outputDirectory, $"Output.png");
            await image.SaveAsync(outputFilename);

            //Unload
            await controlNet.UnloadAsync();
            await pipeline.UnloadAsync();
        }
    }
}

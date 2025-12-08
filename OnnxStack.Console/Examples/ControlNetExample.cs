using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class ControlNetExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public ControlNetExample()
        {
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
            // Execution Provider
            var provider = Providers.DirectML(0);

            // Load Control Image
            var controlImage = await OnnxImage.FromFileAsync("D:\\Repositories\\OnnxStack\\Assets\\Samples\\OpenPose.png");

            // Create ControlNet
            var controlNet = ControlNetModel.Create(provider, "D:\\Models\\controlnet_onnx\\controlnet\\openpose.onnx");

            // Create Pipeline
            var pipeline = StableDiffusionPipeline.CreatePipeline(provider, "D:\\Models\\stable-diffusion-v1-5-onnx");

            // Prompt
            var generateOptions = new GenerateOptions
            {
                Prompt = "Stormtrooper",
                Diffuser = DiffuserType.ControlNet,
                InputContolImage = controlImage,
                ControlNet = controlNet
            };

            // Run pipeline
            var result = await pipeline.RunAsync(generateOptions, progressCallback: OutputHelpers.ProgressCallback);

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

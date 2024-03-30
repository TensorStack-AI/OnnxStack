using OnnxStack.Core.Image;
using OnnxStack.FeatureExtractor.Pipelines;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Pipelines;
using SixLabors.ImageSharp;

namespace OnnxStack.Console.Runner
{
    public sealed class ControlNetFeatureExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public ControlNetFeatureExample(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(ControlNetFeatureExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 12;

        public string Name => "ControlNet + Feature Extraction Example";

        public string Description => "ControlNet StableDiffusion with input image Depth feature extraction";

        /// <summary>
        /// ControlNet Example
        /// </summary>
        public async Task RunAsync()
        {
            // Load Control Image
            var inputImage = await OnnxImage.FromFileAsync("D:\\Repositories\\OnnxStack\\Assets\\Samples\\Img2Img_Start.bmp");

            // Create Annotation pipeline
            var annotationPipeline = FeatureExtractorPipeline.CreatePipeline("D:\\Repositories\\controlnet_onnx\\annotators\\depth.onnx", sampleSize: 512, normalizeOutputTensor: true);

            // Create Depth Image
            var controlImage = await annotationPipeline.RunAsync(inputImage);

            // Save Depth Image (Debug Only)
            await controlImage.SaveAsync(Path.Combine(_outputDirectory, $"Depth.png"));

            // Create ControlNet
            var controlNet = ControlNetModel.Create("D:\\Repositories\\controlnet_onnx\\controlnet\\depth.onnx");

            // Create Pipeline
            var pipeline = StableDiffusionPipeline.CreatePipeline("D:\\Repositories\\stable_diffusion_onnx", ModelType.ControlNet);

            // Prompt
            var promptOptions = new PromptOptions
            {
                Prompt = "steampunk dog",
                DiffuserType = DiffuserType.ControlNet,
                InputContolImage = controlImage
            };

            // Run pipeline
            var result = await pipeline.RunAsync(promptOptions, controlNet: controlNet, progressCallback: OutputHelpers.ProgressCallback);

            // Create Image from Tensor result
            var image = new OnnxImage(result);

            // Save Image File
            var outputFilename = Path.Combine(_outputDirectory, $"Output.png");
            await image.SaveAsync(outputFilename);

            //Unload
            await annotationPipeline.UnloadAsync();
            await controlNet.UnloadAsync();
            await pipeline.UnloadAsync();
        }
    }
}

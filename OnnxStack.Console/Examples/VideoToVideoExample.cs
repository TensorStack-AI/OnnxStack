using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Common;
using OnnxStack.FeatureExtractor.Pipelines;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class VideoToVideoExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public VideoToVideoExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(VideoToVideoExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 4;

        public string Name => "Video To Video Demo";

        public string Description => "Video Stable Diffusion Inference";

        public async Task RunAsync()
        {
            // Execution provider
            var provider = Providers.DirectML(0);

            // Load Video
            var videoInput = await OnnxVideo.FromFileAsync("C:\\Users\\Deven\\Downloads\\FrameMerge\\FrameMerge\\Input.gif");

            // Create Pipeline
            var pipeline = StableDiffusionPipeline.CreatePipeline(provider, "M:\\Models\\stable-diffusion-instruct-pix2pix-onnx", ModelType.Instruct);

            // Create ControlNet
            var controlNet = ControlNetModel.Create(provider, "M:\\Models\\ControlNet-onnx\\SD\\Depth\\model.onnx");

            // Create Feature Extractor
            var featureExtractor = FeatureExtractorPipeline.CreatePipeline(provider, "M:\\Models\\FeatureExtractor-onnx\\SD\\Depth\\model.onnx", sampleSize: 512, normalizeOutputType: Core.Image.ImageNormalizeType.MinMax);

            // Create Depth Image
            var controlVideo = await featureExtractor.RunAsync(videoInput, new FeatureExtractorOptions());

            // GenerateVideoOptions
            var generateOptions = new GenerateOptions
            {
                Prompt = "make a 3D pixar cartoon",
                Diffuser = DiffuserType.ControlNetImage,
                InputVideo = videoInput,
                InputContolVideo = controlVideo,
                ControlNet = controlNet,
                SchedulerOptions = pipeline.DefaultSchedulerOptions with
                {
                    Seed = 302730660,
                    GuidanceScale2 = 1f,
                    ConditioningScale = 0.6f,
                    SchedulerType = SchedulerType.Euler,
                    GuidanceScale = 5f,
                    InferenceSteps = 20,
                    //TimestepSpacing = TimestepSpacingType.Linspace
                    //Width = 512,
                    //Height = 512
                }
            };

            // Run pipeline
            var result = await pipeline.GenerateVideoAsync(generateOptions, progressCallback: OutputHelpers.FrameProgressCallback);

            // Save Video File
            await result.SaveAsync(Path.Combine(_outputDirectory, $"Result_{generateOptions.SchedulerOptions.Seed}.mp4"));

            // Unload
            await pipeline.UnloadAsync();
        }
    }
}

using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Pipelines;
using OnnxStack.StableDiffusion.Config;

namespace OnnxStack.Console.Runner
{
    public sealed class FeatureExtractorVideoExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public FeatureExtractorVideoExample(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(FeatureExtractorVideoExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 13;

        public string Name => "Feature Extractor Video Example";

        public string Description => "Video exmaple using basic feature extractor";

        /// <summary>
        /// ControlNet Example
        /// </summary>
        public async Task RunAsync()
        {
            // Read Video
            var videoFile = "C:\\Users\\Deven\\Pictures\\parrot.mp4";
            var videoInfo = await VideoHelper.ReadVideoInfoAsync(videoFile);

            // Create pipeline
            var pipeline = FeatureExtractorPipeline.CreatePipeline("D:\\Repositories\\controlnet_onnx\\annotators\\canny.onnx");

            // Create Video Stream
            var videoStream = VideoHelper.ReadVideoStreamAsync(videoFile, videoInfo.FrameRate);

            // Create Pipeline Stream
            var pipelineStream = pipeline.RunAsync(videoStream);

            // Write Video Stream
            await VideoHelper.WriteVideoStreamAsync(videoInfo, pipelineStream, Path.Combine(_outputDirectory, $"Result.mp4"));

            //Unload
            await pipeline.UnloadAsync();
        }
    }
}

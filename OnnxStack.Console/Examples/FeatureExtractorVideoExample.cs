using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Common;
using OnnxStack.FeatureExtractor.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class FeatureExtractorVideoExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public FeatureExtractorVideoExample()
        {
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
            // Execution provider
            var provider = Providers.DirectML(0);

            // Read Video
            var videoFile = "C:\\Users\\Deven\\Pictures\\parrot.mp4";
            var videoInfo = await VideoHelper.ReadVideoInfoAsync(videoFile);

            // Create pipeline
            var pipeline = FeatureExtractorPipeline.CreatePipeline(provider, "D:\\Repositories\\controlnet_onnx\\annotators\\canny.onnx");

            // Create Video Stream
            var videoStream = VideoHelper.ReadVideoStreamAsync(videoFile);

            // Create Pipeline Stream
            var pipelineStream = pipeline.RunAsync(videoStream, new FeatureExtractorOptions());

            // Write Video Stream
            await VideoHelper.WriteVideoStreamAsync(Path.Combine(_outputDirectory, $"Result.mp4"), pipelineStream, videoInfo.FrameRate, videoInfo.Width, videoInfo.Height);

            //Unload
            await pipeline.UnloadAsync();
        }
    }
}

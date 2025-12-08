using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Pipelines;
using OnnxStack.ImageUpscaler.Common;

namespace OnnxStack.Console.Runner
{
    public sealed class UpscaleStreamExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public UpscaleStreamExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(UpscaleStreamExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 10;

        public string Name => "Upscale Video Streaming Demo";

        public string Description => "Upscales a video stream";

        public async Task RunAsync()
        {
            // Execution provider
            var provider = Providers.DirectML(0);

            // Read Video
            var videoFile = "C:\\Users\\Deven\\Pictures\\parrot.mp4";
            var videoInfo = await VideoHelper.ReadVideoInfoAsync(videoFile);

            // Create pipeline
            var pipeline = ImageUpscalePipeline.CreatePipeline(provider, "D:\\Repositories\\upscaler\\SwinIR\\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx", 4, 512);

            // Load pipeline
            await pipeline.LoadAsync();

            // Create Video Stream
            var videoStream = VideoHelper.ReadVideoStreamAsync(videoFile);

            // Create Pipeline Stream
            var pipelineStream = pipeline.RunAsync(videoStream, new UpscaleOptions());

            // Write Video Stream
            await VideoHelper.WriteVideoStreamAsync(Path.Combine(_outputDirectory, $"Result.mp4"), pipelineStream, videoInfo.FrameRate, videoInfo.Width, videoInfo.Height);

            //Unload
            await pipeline.UnloadAsync();
        }

    }
}

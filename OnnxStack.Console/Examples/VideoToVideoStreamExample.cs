using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class VideoToVideoStreamExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public VideoToVideoStreamExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(VideoToVideoStreamExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 4;

        public string Name => "Video To Video Stream Demo";

        public string Description => "Video Stream Stable Diffusion Inference";

        public async Task RunAsync()
        {
            // Execution provider
            var provider = Providers.DirectML(0);

            // Read Video
            var videoFile = "C:\\Users\\Deven\\Pictures\\gidsgphy.gif";
            var videoInfo = await VideoHelper.ReadVideoInfoAsync(videoFile);

            // Create Pipeline
            var pipeline = StableDiffusionPipeline.CreatePipeline(provider, "M:\\Models\\stable-diffusion-v1-5-onnx");
            OutputHelpers.WriteConsole($"Loading Model `{pipeline.Name}`...", ConsoleColor.Cyan);

            // Add text and video to prompt
            var generateOptions = new GenerateOptions
            {
                Prompt = "Iron Man",
                Diffuser = DiffuserType.ImageToImage
            };

            // Create Video Stream
            var videoStream = VideoHelper.ReadVideoStreamAsync(videoFile, videoInfo.FrameRate, 512, 512);

            // Create Pipeline Stream
            var pipelineStream = pipeline.GenerateVideoStreamAsync(generateOptions, videoStream, progressCallback: OutputHelpers.ProgressCallback);

            // Write Video Stream
            await VideoHelper.WriteVideoStreamAsync(Path.Combine(_outputDirectory, $"{pipeline.PipelineType}.mp4"), pipelineStream, videoInfo.FrameRate, videoInfo.Width, videoInfo.Height);

            //Unload
            await pipeline.UnloadAsync();
        }
    }
}

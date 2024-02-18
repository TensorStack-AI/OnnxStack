using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Pipelines;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class VideoToVideoStreamExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public VideoToVideoStreamExample(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(VideoToVideoStreamExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 4;

        public string Name => "Video To Video Stream Demo";

        public string Description => "Video Stream Stable Diffusion Inference";

        public async Task RunAsync()
        {

            // Read Video
            var videoFile = "C:\\Users\\Deven\\Pictures\\gidsgphy.gif";
            var videoInfo = await VideoHelper.ReadVideoInfoAsync(videoFile);

            // Loop though the appsettings.json model sets
            foreach (var modelSet in _configuration.ModelSets)
            {
                OutputHelpers.WriteConsole($"Loading Model `{modelSet.Name}`...", ConsoleColor.Cyan);

                // Create Pipeline
                var pipeline = PipelineBase.CreatePipeline(modelSet);

                // Preload Models (optional)
                await pipeline.LoadAsync();

                // Add text and video to prompt
                var promptOptions = new PromptOptions
                {
                    Prompt = "Iron Man",
                    DiffuserType = DiffuserType.ImageToImage
                };


                // Create Video Stream
                var videoStream = VideoHelper.ReadVideoStreamAsync(videoFile, videoInfo.FrameRate);

                // Create Pipeline Stream
                var pipelineStream = pipeline.GenerateVideoStreamAsync(videoStream, promptOptions, progressCallback:OutputHelpers.ProgressCallback);

                // Write Video Stream
                await VideoHelper.WriteVideoStreamAsync(videoInfo, pipelineStream, Path.Combine(_outputDirectory, $"{modelSet.PipelineType}.mp4"));

                //Unload
                await pipeline.UnloadAsync();
            }
        }
    }
}

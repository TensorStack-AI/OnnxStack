using OnnxStack.Core.Services;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class VideoToVideoExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;
        private readonly IVideoService _videoService;

        public VideoToVideoExample(StableDiffusionConfig configuration, IVideoService videoService)
        {
            _configuration = configuration;
            _videoService = videoService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(VideoToVideoExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public string Name => "Video To Video Demo";

        public string Description => "Video Stable Diffusion Inference";

        public async Task RunAsync()
        {
            string inputVideoPath = "C:\\Users\\Deven\\Pictures\\gidsgphy.gif";
            var inputFile = File.ReadAllBytes(inputVideoPath);
            var videoInfo = await _videoService.GetVideoInfoAsync(inputFile);
            var videoInput = await _videoService.CreateFramesAsync(inputFile, videoInfo.FPS);

            // Progress Callback (optional)
            var progressCallback = (DiffusionProgress progress) => OutputHelpers.WriteConsole($"Frame: {progress.BatchValue}/{progress.BatchMax} - Step: {progress.StepValue}/{progress.StepMax}", ConsoleColor.Cyan);


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
                    DiffuserType = DiffuserType.ImageToImage,
                    InputVideo = new VideoInput(videoInput)
                };

                // Run pipeline
                var result = await pipeline.RunAsync(promptOptions, progressCallback: progressCallback);

                // Create Video from Tensor result
                var videoResult = await _videoService.CreateVideoAsync(result, videoInfo.FPS);

                // Save Video File
                var outputFilename = Path.Combine(_outputDirectory, $"{modelSet.Name}.mp4");
                await File.WriteAllBytesAsync(outputFilename, videoResult.Data);
            }
        }
    }
}

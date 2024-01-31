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

        public int Index => 4;

        public string Name => "Video To Video Demo";

        public string Description => "Video Stable Diffusion Inference";

        public async Task RunAsync()
        {
            // Load Video
            var targetFPS = 15;
            var videoInput = await VideoInput.FromFileAsync("C:\\Users\\Deven\\Pictures\\gidsgphy.gif", targetFPS);

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
                    InputVideo = videoInput
                };

                // Run pipeline
                var result = await pipeline.RunAsync(promptOptions, progressCallback: OutputHelpers.FrameProgressCallback);

                // Save Video File
                var outputFilename = Path.Combine(_outputDirectory, $"{modelSet.Name}.mp4");
                await VideoInput.SaveFileAsync(result, outputFilename, targetFPS);
            }
        }
    }
}

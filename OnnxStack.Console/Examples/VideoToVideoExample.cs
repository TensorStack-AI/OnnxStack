using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class VideoToVideoExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;

        public VideoToVideoExample(StableDiffusionConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(VideoToVideoExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 4;

        public string Name => "Video To Video Demo";

        public string Description => "Video Stable Diffusion Inference";

        public async Task RunAsync()
        {
            // Load Video
            var videoInput = await OnnxVideo.FromFileAsync("C:\\Users\\Deven\\Pictures\\gidsgphy.gif");

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
                var result = await pipeline.GenerateVideoAsync(promptOptions, progressCallback: OutputHelpers.FrameProgressCallback);

                // Save Video File
                await result.SaveAsync(Path.Combine(_outputDirectory, $"Result.mp4"));
            }
        }
    }
}

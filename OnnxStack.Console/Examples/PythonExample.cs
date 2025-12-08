using Microsoft.Extensions.Logging;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Python;
using OnnxStack.StableDiffusion.Python.Config;
using OnnxStack.StableDiffusion.Python.Pipelines;

namespace OnnxStack.Console.Runner
{
    public sealed class PythonExample : IExampleRunner
    {
        private readonly ILogger _logger;
        private readonly string _outputDirectory;

        public PythonExample(ILogger<PythonExample> logger)
        {
            _logger = logger;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(PythonExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 101;

        public string Name => "Python Pipeline Example";

        public string Description => "Run python pipelines";

        public async Task RunAsync()
        {
            // Download and Install
            await PythonEnvironment.DownloadAsync();

            // Create Environment
            await PythonEnvironment.CreateAsync("cuda", logger: _logger);
            //await PythonEnvironment.RecreateAsync("cuda", logger: _logger);

            // Config
            var config = new PythonModelConfig
            {
                Device = "cuda",
                DeviceId = 0,
                ModelPath = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                EnableModelCpuOffload = true,
                EnableSequentialCpuOffload = false,
                EnableVaeSlicing = false,
                EnableVaeTiling = false,
            };

            // Create Pipeline
            var pipeline = DiffusersPipeline.CreatePipeline(config, _logger);

            // Generate options
            var generateOptions = new GenerateOptions
            {
                Prompt = "Beautiful, snowy Tokyo city is bustling. The camera moves through the bustling city street, following several people enjoying the beautiful snowy weather and shopping at nearby stalls",
                NegativePrompt = "painting, drawing, sketches, illustration, cartoon, crayon",
                Diffuser = DiffuserType.TextToVideo,

                MotionFrames = 81,
                OutputFrameRate = 15,
                SchedulerOptions = new SchedulerOptions
                {
                    Width = 832,
                    Height = 480,
                    InferenceSteps = 50,
                    GuidanceScale = 5f
                }
            };

            // Run pipeline
            var result = await pipeline.GenerateVideoAsync(generateOptions, progressCallback: OutputHelpers.PythonProgressCallback);

            // Save File
            await result.SaveAsync(Path.Combine(_outputDirectory, $"{generateOptions.SchedulerOptions.Seed}.mp4"));

            //Unload
            await pipeline.UnloadAsync();
        }
    }
}

using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDebug : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly IStableDiffusionService _stableDiffusionService;

        public StableDebug(IStableDiffusionService stableDiffusionService)
        {
            _stableDiffusionService = stableDiffusionService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDebug));
        }

        public string Name => "Stable Diffusion Debug";

        public string Description => "Stable Diffusion Debugger";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);

            var prompt = "an apple wearing a hat";
            while (true)
            {
                var options = new StableDiffusionOptions
                {
                    Prompt = prompt,
                    //Seed = 42069
                    Seed = Random.Shared.Next()
                };
                foreach (var schedulerType in Enum.GetValues<SchedulerType>())
                {
                    options.SchedulerType = schedulerType;
                    OutputHelpers.WriteConsole("Generating Image...", ConsoleColor.Green);
                    await GenerateImage(options);
                }
            }

        }


        private async Task<bool> GenerateImage(StableDiffusionOptions options)
        {
            var timestamp = Stopwatch.GetTimestamp();
            var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{options.SchedulerType}.png");
            if (await _stableDiffusionService.TextToImageFile(options, outputFilename))
            {
                OutputHelpers.WriteConsole($"{options.SchedulerType} Image Created, FilePath: {outputFilename}", ConsoleColor.Green);
                OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);
                return true;
            }

            OutputHelpers.WriteConsole($"Failed to create image", ConsoleColor.Red);
            return false;
        }
    }
}

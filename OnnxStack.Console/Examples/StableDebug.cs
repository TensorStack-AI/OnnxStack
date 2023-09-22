using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Services;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public class StableDebug : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly OnnxStackConfig _configuration;

        public StableDebug(OnnxStackConfig configuration)
        {
            _configuration = configuration;
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
            using (var stableDiffusionService = new StableDiffusionService(_configuration))
            {
                while (true)
                {
                    var options = new StableDiffusionOptions
                    {
                        Prompt = prompt
                    };
                    foreach (var schedulerType in Enum.GetValues<SchedulerType>())
                    {
                        options.SchedulerType = schedulerType;

                        OutputHelpers.WriteConsole("Generating Image...", ConsoleColor.Green);
                        await GenerateImage(stableDiffusionService, options);
                    }
                }
            }
        }


        private async Task<bool> GenerateImage(IStableDiffusionService stableDiffusionService, StableDiffusionOptions options)
        {
            var timestamp = Stopwatch.GetTimestamp();
            var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{options.SchedulerType}.png");
            if (await stableDiffusionService.TextToImageFile(options, outputFilename))
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

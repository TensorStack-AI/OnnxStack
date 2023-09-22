using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Services;

namespace OnnxStack.Console.Runner
{
    public class StableDiffusionExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly OnnxStackConfig _configuration;

        public StableDiffusionExample(OnnxStackConfig configuration)
        {
            _configuration = configuration;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionExample));
        }

        public string Name => "Stable Diffusion Demo";

        public string Description => "Creates images from the text prompt provided using all Scheduler types";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);

            using (var stableDiffusionService = new StableDiffusionService(_configuration))
            {
                while (true)
                {
                    OutputHelpers.WriteConsole("Please type a prompt and press ENTER", ConsoleColor.Yellow);
                    var prompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                    OutputHelpers.WriteConsole("Please type a negative prompt and press ENTER (optional)", ConsoleColor.Yellow);
                    var negativePrompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                    var options = new StableDiffusionOptions
                    {
                        Prompt = prompt,
                        NegativePrompt = negativePrompt
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
            var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{options.SchedulerType}.png");
            if (await stableDiffusionService.TextToImageFile(options, outputFilename))
            {
                OutputHelpers.WriteConsole($"{options.SchedulerType} Image Created, FilePath: {outputFilename}", ConsoleColor.Green);
                return true;
            }
            return false;
        }
    }
}

using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Services;

namespace OnnxStack.Console.Runner
{
    public class StableDiffusionExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly IStableDiffusionService _stableDiffusionService;

        public StableDiffusionExample(IStableDiffusionService stableDiffusionService)
        {
            _stableDiffusionService = stableDiffusionService;
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

            while (true)
            {
                OutputHelpers.WriteConsole("Please type a prompt and press ENTER", ConsoleColor.Yellow);
                var prompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                OutputHelpers.WriteConsole("Please type a negative prompt and press ENTER (optional)", ConsoleColor.Yellow);
                var negativePrompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                var options = new StableDiffusionOptions
                {
                    Prompt = prompt,
                    NegativePrompt = negativePrompt,
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
            var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{options.SchedulerType}.png");
            if (await _stableDiffusionService.TextToImageFile(options, outputFilename))
            {
                OutputHelpers.WriteConsole($"{options.SchedulerType} Image Created, FilePath: {outputFilename}", ConsoleColor.Green);
                return true;
            }
            return false;
        }
    }
}

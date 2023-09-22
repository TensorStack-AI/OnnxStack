using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Services;

namespace OnnxStack.Console.Runner
{
    public class StableDiffusionExample : IExampleRunner
    {
        private readonly OnnxStackConfig _configuration;

        public StableDiffusionExample(OnnxStackConfig configuration)
        {
            _configuration = configuration;
        }

        public string Name => "Stable Diffusion Demo";

        public string Description => "Creates images from the text prompt provided using all Scheduler types";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            while (true)
            {
                OutputHelpers.WriteConsole("Please type a prompt and press ENTER", ConsoleColor.Yellow);
                var prompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

                OutputHelpers.WriteConsole("Please type a negative prompt and press ENTER (optional)", ConsoleColor.Yellow);
                var negativePrompt = OutputHelpers.ReadConsole(ConsoleColor.Cyan);
                using (var stableDiffusionService = new StableDiffusionService(_configuration))
                {
                    // Generate image using LMSScheduler
                    await GenerateImage(stableDiffusionService, prompt, negativePrompt, SchedulerType.LMSScheduler);

                    // Generate image using EulerAncestralScheduler
                    await GenerateImage(stableDiffusionService, prompt, negativePrompt, SchedulerType.EulerAncestralScheduler);
                }
            }
        }

        private async Task<bool> GenerateImage(IStableDiffusionService stableDiffusionService, string prompt, string negativePrompt, SchedulerType schedulerType)
        {
            var outputPath = Path.Combine(Directory.GetCurrentDirectory(), $"{schedulerType}_{DateTime.Now.ToString("yyyyMMddHHmmSS")}.png");
            var schedulerConfig = new SchedulerConfig
            {
                SchedulerType = schedulerType
            };
            if (await stableDiffusionService.TextToImageFile(prompt, negativePrompt, outputPath, schedulerConfig))
            {
                OutputHelpers.WriteConsole($"{schedulerType} Image Created, FilePath: {outputPath}", ConsoleColor.Green);
                return true;
            }
            return false;
        }
    }
}

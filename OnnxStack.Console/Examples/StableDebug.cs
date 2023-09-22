using OnnxStack.Core.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Services;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public class StableDebug : IExampleRunner
    {
        private readonly OnnxStackConfig _configuration;

        public StableDebug(OnnxStackConfig configuration)
        {
            _configuration = configuration;
        }

        public string Name => "Stable Diffusion Debug";

        public string Description => "Stable Diffusion Debugger";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            var prompt = "an apple wearing a hat";
            using (var stableDiffusionService = new StableDiffusionService(_configuration))
            {
                while (true)
                {
                    // Generate image using LMSScheduler
                    await GenerateImage(stableDiffusionService, prompt, null, SchedulerType.LMSScheduler);


                    // Generate image using EulerAncestralScheduler
                    await GenerateImage(stableDiffusionService, prompt, null, SchedulerType.EulerAncestralScheduler);
                }
            }
        }

        private async Task<bool> GenerateImage(IStableDiffusionService stableDiffusionService, string prompt, string negativePrompt, SchedulerType schedulerType)
        {
            var timestamp = Stopwatch.GetTimestamp();
            var outputPath = Path.Combine(Directory.GetCurrentDirectory(), "Images", $"{schedulerType}_{DateTime.Now.ToString("yyyyMMddHHmmSS")}.png");
            var schedulerConfig = new SchedulerConfig
            {
                SchedulerType = schedulerType
            };
            if (await stableDiffusionService.TextToImageFile(prompt, negativePrompt, outputPath, schedulerConfig))
            {
                OutputHelpers.WriteConsole($"{schedulerType} Image Created, FilePath: {outputPath}", ConsoleColor.Green);
                OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);
                return true;
            }


            return false;
        }
    }
}

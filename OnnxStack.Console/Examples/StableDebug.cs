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

            var prompt = "High-fashion photography in an abandoned industrial warehouse, with dramatic lighting and edgy outfits, detailed clothing, intricate clothing, seductive pose, action pose, motion, beautiful digital artwork, atmospheric, warm sunlight, photography, neo noir, bokeh, beautiful dramatic lighting, shallow depth of field, photorealism, volumetric lighting, Ultra HD, raytracing, studio quality, octane render";
            var negativePrompt = "painting, drawing, sketches, monochrome, grayscale, illustration, anime, cartoon, graphic, text, crayon, graphite, abstract, easynegative, low quality, normal quality, worst quality, lowres, close up, cropped, out of frame, jpeg artifacts, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, glitch, deformed, mutated, cross-eyed, ugly, dehydrated, bad anatomy, bad proportions, gross proportions, cloned face, disfigured, malformed limbs, missing arms, missing legs fused fingers, too many fingers,extra fingers, extra limbs,, extra arms, extra legs,disfigured,";
            while (true)
            {
                var promptOptions = new PromptOptions
                {
                    Prompt = prompt,
                    NegativePrompt = negativePrompt,
                    SchedulerType = SchedulerType.LMSScheduler
                    //InputImage = @"C:\Users\Deven\Pictures\sketch-mountains-input.jpg"
                };

                var schedulerOptions = new SchedulerOptions
                {
                    Seed = 624461087,
                    //Seed = Random.Shared.Next(),
                    GuidanceScale = 8,
                    InferenceSteps = 22,
                    //StepsOffset = 1
                };

                foreach (var schedulerType in Enum.GetValues<SchedulerType>())
                {
                    promptOptions.SchedulerType = schedulerType;
                    OutputHelpers.WriteConsole("Generating Image...", ConsoleColor.Green);
                    await GenerateImage(promptOptions, schedulerOptions);
                }
            }
        }


        private async Task<bool> GenerateImage(PromptOptions prompt, SchedulerOptions options)
        {
            var timestamp = Stopwatch.GetTimestamp();
            var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{prompt.SchedulerType}.png");
            var result = await _stableDiffusionService.TextToImageFile(prompt, options, outputFilename);
            if (result is not null)
            {
                OutputHelpers.WriteConsole($"{prompt.SchedulerType} Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
                OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);
                return true;
            }

            OutputHelpers.WriteConsole($"Failed to create image", ConsoleColor.Red);
            return false;
        }
    }
}

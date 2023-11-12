using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Services;
using SixLabors.ImageSharp;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class UpscaleDebug : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly IStableDiffusionService _stableDiffusionService;

        public UpscaleDebug(IStableDiffusionService stableDiffusionService)
        {
            _stableDiffusionService = stableDiffusionService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(UpscaleDebug));
        }

        public string Name => "Upscale Debug";

        public string Description => "Upscale Debugger";


        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);

            var model = _stableDiffusionService.Models.FirstOrDefault();

            var promptOptions = new PromptOptions
            {
                Prompt = "a white cat",
     
                InputImage = new StableDiffusion.Models.InputImage(File.ReadAllBytes(@"C:\Users\Deven\Pictures\1low_res_cat.png")),
                DiffuserType = DiffuserType.ImageUpscale
            };

            var schedulerOptions = new SchedulerOptions
            {
                Seed = 624461087,
                //Seed = Random.Shared.Next(),
                GuidanceScale = 0,
                InferenceSteps = 100,
                SchedulerType = SchedulerType.EulerAncestral,
                NoiseLevel = 20,
                Width = 128,
                Height = 128
            };

            OutputHelpers.WriteConsole($"Loading Model `{model.Name}`...", ConsoleColor.Green);
            await _stableDiffusionService.LoadModel(model);

           // foreach (var schedulerType in Helpers.GetPipelineSchedulers(model.PipelineType))
            {

                OutputHelpers.WriteConsole($"Generating {schedulerOptions.SchedulerType} Image...", ConsoleColor.Green);
                await GenerateImage(model, promptOptions, schedulerOptions);
            }

            OutputHelpers.WriteConsole($"Unloading Model `{model.Name}`...", ConsoleColor.Green);
            await _stableDiffusionService.UnloadModel(model);
        }


        private async Task<bool> GenerateImage(ModelOptions model, PromptOptions prompt, SchedulerOptions options)
        {
            var timestamp = Stopwatch.GetTimestamp();
            var outputFilename = Path.Combine(_outputDirectory, $"{options.Seed}_{options.SchedulerType}.png");
            var result = await _stableDiffusionService.GenerateAsImageAsync(model, prompt, options);
            if (result is not null)
            {
                await result.SaveAsPngAsync(outputFilename);
                OutputHelpers.WriteConsole($"{options.SchedulerType} Image Created: {Path.GetFileName(outputFilename)}", ConsoleColor.Green);
                OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);
                return true;
            }

            OutputHelpers.WriteConsole($"Failed to create image", ConsoleColor.Red);
            return false;
        }
    }
}

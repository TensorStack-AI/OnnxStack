using OnnxStack.StableDiffusion;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Gif;
using SixLabors.ImageSharp.PixelFormats;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionGif : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly IStableDiffusionService _stableDiffusionService;

        public StableDiffusionGif(IStableDiffusionService stableDiffusionService)
        {
            _stableDiffusionService = stableDiffusionService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionGif));
        }

        public string Name => "Stable Diffusion Gif";

        public string Description => "Stable Diffusion Gif";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            Directory.CreateDirectory(_outputDirectory);

            var prompt = "elon musk wearing a red hat and sunglasses";
            var negativePrompt = "";

            var promptOptions = new PromptOptions
            {
                Prompt = prompt,
                NegativePrompt = negativePrompt,
                DiffuserType = DiffuserType.ImageToImage
            };

            var schedulerOptions = new SchedulerOptions
            {
                SchedulerType = SchedulerType.LCM,
                Seed = 624461087,
                GuidanceScale = 1f,
                InferenceSteps = 20,
                Strength = 0.3f,
                 
            };

            // Choose Model
            var model = _stableDiffusionService.Models.FirstOrDefault(x => x.Name == "LCM-Dreamshaper-V7");
            OutputHelpers.WriteConsole($"Loading Model `{model.Name}`...", ConsoleColor.Green);
            await _stableDiffusionService.LoadModelAsync(model);

            // Set Size
            schedulerOptions.Width = model.SampleSize;
            schedulerOptions.Height = model.SampleSize;


            using Image<Rgba32> gifDestination = new(schedulerOptions.Width, schedulerOptions.Height);
            {
                var gifMetaData = gifDestination.Metadata.GetGifMetadata();
                gifMetaData.RepeatCount = 0; // Loop

                using (var gifSource = await Image.LoadAsync(Path.Combine(_outputDirectory, "source.gif")))
                {
                    for (int i = 0; i < gifSource.Frames.Count; i++)
                    {
                        promptOptions.InputImage = new InputImage(gifSource.Frames.CloneFrame(i).CloneAs<Rgba32>());
                        var result = await _stableDiffusionService.GenerateAsImageAsync(model, promptOptions, schedulerOptions);



                        gifDestination.Frames.AddFrame(result.Frames.RootFrame);

                        OutputHelpers.WriteConsole($"Frame: {i + 1}/{gifSource.Frames.Count}", ConsoleColor.Cyan);
                    }

                    var outputFilename = Path.Combine(_outputDirectory, $"result.gif");
                    await gifDestination.SaveAsGifAsync(outputFilename);
                }
            }
        }

    }
}

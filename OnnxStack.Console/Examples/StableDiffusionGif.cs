using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace OnnxStack.Console.Runner
{
    public sealed class StableDiffusionGif : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;
        private readonly IStableDiffusionService _stableDiffusionService;

        public StableDiffusionGif(StableDiffusionConfig configuration, IStableDiffusionService stableDiffusionService)
        {
            _configuration = configuration;
            _stableDiffusionService = stableDiffusionService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(StableDiffusionGif));
            Directory.CreateDirectory(_outputDirectory);
        }

        public string Name => "Stable Diffusion Gif";

        public string Description => "Stable Diffusion Gif";

        /// <summary>
        /// Stable Diffusion Demo
        /// </summary>
        public async Task RunAsync()
        {
            var prompt = "Iron Man";
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
                //  Seed = 624461087,
                GuidanceScale = 1f,
                InferenceSteps = 20,
                Strength = 0.35f,
            };

            // Choose Model
            var model = _configuration.ModelSets.FirstOrDefault(x => x.Name == "LCM-Dreamshaper-V7");
            OutputHelpers.WriteConsole($"Loading Model `{model.Name}`...", ConsoleColor.Green);
            await _stableDiffusionService.LoadModelAsync(model);

            // Set Size
            schedulerOptions.Width = model.SampleSize;
            schedulerOptions.Height = model.SampleSize;

            var repeatCount = 0;
            var frameDelay = 0;
            var imageMerge = 0.88f;

            using Image<Rgba32> gifDestination = new(schedulerOptions.Width, schedulerOptions.Height);
            {
                var gifMetaData = gifDestination.Metadata.GetGifMetadata();
                gifMetaData.RepeatCount = (ushort)repeatCount; // Loop

                using (var gifSource = await Image.LoadAsync(Path.Combine(_outputDirectory, "Source.gif")))
                using (var frame = gifSource.Frames.CloneFrame(0))
                {
                    frame.Mutate(x => x.Resize(gifDestination.Size));
                    for (int i = 0; i < gifSource.Frames.Count; i++)
                    {
                        var newFrame = gifSource.Frames.CloneFrame(i);
                        newFrame.Mutate(x => x.Resize(gifDestination.Size));

                        // Draw each frame on top of the last to fix issues with compressed gifs
                        frame.Mutate(x => x.DrawImage(newFrame, 1f));

                        var mergedFrame = frame.CloneAs<Rgba32>();
                        if (i > 1 && imageMerge < 1)
                        {
                            // Get Previous SD Frame
                            var previousFrame = gifDestination.Frames.CloneFrame(i - 1);

                            // Set to Grayscale
                            previousFrame.Mutate(x => x.Grayscale());

                            // Set BG Frame Opacity
                            mergedFrame.Mutate(x => x.Opacity(imageMerge));

                            // Draw Previous SD Frame
                            mergedFrame.Mutate(x => x.DrawImage(previousFrame, 1f - imageMerge));
                        }


                        // Save Debug Output
                        await mergedFrame.SaveAsPngAsync(Path.Combine(_outputDirectory, $"Debug-Frame.png"));

                        // Set prompt Image, Run Diffusion
                        promptOptions.InputImage = new InputImage(mergedFrame.CloneAs<Rgba32>());
                        var result = await _stableDiffusionService.GenerateAsImageAsync(new ModelOptions(model), promptOptions, schedulerOptions);

                        // Save Debug Output
                        await result.SaveAsPngAsync(Path.Combine(_outputDirectory, $"Debug-Output.png"));

                        // Set Fraem metadata
                        var metadata = result.Frames.RootFrame.Metadata.GetGifMetadata();
                        metadata.FrameDelay = frameDelay;

                        // Add Result to Gif
                        gifDestination.Frames.InsertFrame(i, result.Frames.RootFrame);
                        OutputHelpers.WriteConsole($"Frame: {i + 1}/{gifSource.Frames.Count}", ConsoleColor.Cyan);
                    }
                }

                // Save Result
                await gifDestination.SaveAsGifAsync(Path.Combine(_outputDirectory, $"Result.gif"));

            }
        }

    }
}

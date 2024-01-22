using OnnxStack.Console;
using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using OnnxStack.Core.Video;
using OnnxStack.Core.Services;

namespace OnnxStack.Console.Runner
{
    public sealed class VideoToVideoExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;
        private readonly IStableDiffusionService _stableDiffusionService;
        private readonly IVideoService _videoService;

        public VideoToVideoExample(StableDiffusionConfig configuration, IStableDiffusionService stableDiffusionService, IVideoService videoService)
        {
            _configuration = configuration;
            _stableDiffusionService = stableDiffusionService;
            _videoService = videoService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(UpscaleExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public string Name => "Video To Video Demo";

        public string Description => "Vidio Stable Diffusion Inference";

        public async Task RunAsync()
        {
            var model = _configuration.ModelSets.FirstOrDefault(x => x.Name == "LCM-Dreamshaper-V7");
            OutputHelpers.WriteConsole("Loading Model...", ConsoleColor.Cyan);
            await _stableDiffusionService.LoadModelAsync(model);
            OutputHelpers.WriteConsole("Model Loaded.", ConsoleColor.Cyan);
            string inputVideoPath = "C:\\Users\\Hex\\Downloads\\doit.mp4";
            string outputVideoPath = "C:\\Users\\Hex\\Downloads\\doitdiffused.mp4";


            var prompt = "Iron Man";
            var negativePrompt = "";

            var inputVideo = File.ReadAllBytes(inputVideoPath);

            var promptOptions = new PromptOptions
            {
                Prompt = prompt,
                NegativePrompt = negativePrompt,
                DiffuserType = DiffuserType.ImageToImage,
                InputVideo = new VideoInput(inputVideo)
            };

            var schedulerOptions = new SchedulerOptions
            {
                SchedulerType = SchedulerType.LCM,
                GuidanceScale = 1f,
                InferenceSteps = 10,
                Strength = 0.35f,
                Height = 512,
                Width = 512
            };

            var result = await _stableDiffusionService.GenerateAsBytesAsync(new ModelOptions(model), promptOptions, schedulerOptions);
            File.WriteAllBytes(outputVideoPath, result);
        }
    }
}

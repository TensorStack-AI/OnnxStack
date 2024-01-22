using OnnxStack.Core.Image;
using OnnxStack.ImageUpscaler.Config;
using OnnxStack.ImageUpscaler.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.Console.Runner
{
    public sealed class VideoToVideoExampleFromScratch : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly StableDiffusionConfig _configuration;
        private readonly IStableDiffusionService _stableDiffusionService;

        public VideoToVideoExampleFromScratch(StableDiffusionConfig configuration, IStableDiffusionService stableDiffusionService)
        {
            _configuration = configuration;
            _stableDiffusionService = stableDiffusionService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(UpscaleExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public string Name => "Video To Video Demo From Scratch";

        public string Description => "Vidio Stable Diffusion Inference From Scratch";

        public async Task RunAsync()
        {
            var model = _configuration.ModelSets.FirstOrDefault(x => x.Name == "LCM-Dreamshaper-V7");
            OutputHelpers.WriteConsole("Loading Model...", ConsoleColor.Cyan);
            await _stableDiffusionService.LoadModelAsync(model);
            OutputHelpers.WriteConsole("Model Loaded.", ConsoleColor.Cyan);
            string inputVideoPath = "C:\\Users\\Hex\\Downloads\\doit.mp4";
            string outputFramesPath = "C:\\Users\\Hex\\Desktop\\frames\\frame_%04d.png";
            string ffmpegCommand = $"-i \"{inputVideoPath}\" -vf fps=30 -c:v png -y \"{outputFramesPath}\"";
            string ffmpeg = @"C:\Users\Hex\Desktop\OnnxStack\ffmpeg.exe";

            Process process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = ffmpeg,
                    Arguments = ffmpegCommand,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.OutputDataReceived += (sender, e) => OutputHelpers.WriteConsole(e.Data, ConsoleColor.Cyan);
            process.ErrorDataReceived += (sender, e) => OutputHelpers.WriteConsole(e.Data, ConsoleColor.Cyan);

            process.Start();

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            process.WaitForExit();

            outputFramesPath = outputFramesPath.Replace("\\frame_%04d.png", "");

            string[] files = Directory.GetFiles(outputFramesPath);

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
                GuidanceScale = 1f,
                InferenceSteps = 10,
                Strength = 0.35f,
                Height = 512,
                Width = 512
            };

            foreach (string filePath in files)
            {
                OutputHelpers.WriteConsole($"Defusing {filePath}", ConsoleColor.Cyan);
                Image<Rgba32> frameDestination = new(schedulerOptions.Width, schedulerOptions.Height);
                var frameSource = await Image.LoadAsync(filePath);
                frameSource.Mutate(x => x.Resize(frameDestination.Size));
                promptOptions.InputImage = new InputImage(frameSource.CloneAs<Rgba32>());
                frameDestination = await _stableDiffusionService.GenerateAsImageAsync(new ModelOptions(model), promptOptions, schedulerOptions);
                await frameDestination.SaveAsPngAsync(filePath);
                OutputHelpers.WriteConsole($"{filePath} saved", ConsoleColor.Cyan);
            }

            string outputVideoPath = "C:\\Users\\Hex\\Downloads\\doitdefused.mp4";

            ffmpegCommand = $"ffmpeg -framerate 30 -i \"{outputFramesPath}\\frame_%04d.png\" -c:v libx264 -pix_fmt yuv420p -y \"{outputVideoPath}\"";

            // Create a process to run the FFmpeg command
            process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = ffmpeg,
                    Arguments = ffmpegCommand,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.OutputDataReceived += (sender, e) => OutputHelpers.WriteConsole(e.Data, ConsoleColor.Cyan);
            process.ErrorDataReceived += (sender, e) => OutputHelpers.WriteConsole(e.Data, ConsoleColor.Cyan);

            // Start the process
            process.Start();

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            // Wait for the process to exit
            process.WaitForExit();
        }
    }
}
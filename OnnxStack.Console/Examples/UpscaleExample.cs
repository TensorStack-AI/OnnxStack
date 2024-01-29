using OnnxStack.Core.Image;
using OnnxStack.ImageUpscaler.Config;
using OnnxStack.ImageUpscaler.Services;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxStack.Console.Runner
{
    public sealed class UpscaleExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly ImageUpscalerConfig _configuration;
        private readonly IUpscaleService _imageUpscaleService;


        public UpscaleExample(ImageUpscalerConfig configuration, IUpscaleService imageUpscaleService)
        {
            _configuration = configuration;
            _imageUpscaleService = imageUpscaleService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(UpscaleExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 10;

        public string Name => "Image Upscale Demo";

        public string Description => "Upscales images";

        public async Task RunAsync()
        {
            var modelSet = _configuration.ModelSets.FirstOrDefault(x => x.Name == "RealSR BSRGAN x4");



            OutputHelpers.WriteConsole("Enter Image Path", ConsoleColor.Yellow);
            var imageFile = OutputHelpers.ReadConsole(ConsoleColor.Gray);
            if (!File.Exists(imageFile))
            {
                OutputHelpers.WriteConsole("File not found!", ConsoleColor.Red);
                return;
            }

            OutputHelpers.WriteConsole("Loading Model...", ConsoleColor.Cyan);
            await _imageUpscaleService.LoadModelAsync(modelSet);
            OutputHelpers.WriteConsole("Model Loaded.", ConsoleColor.Cyan);

            var inputImage = await Image.LoadAsync<Rgba32>(imageFile);

            OutputHelpers.WriteConsole("Upscaling Image...", ConsoleColor.Cyan);
            var result = await _imageUpscaleService.GenerateAsImageAsync(modelSet, new InputImage(inputImage));
            await result.SaveAsPngAsync(Path.Combine(_outputDirectory, "Result.png"));
            OutputHelpers.WriteConsole("Upscaling Complete.", ConsoleColor.Cyan);
        }

    }
}

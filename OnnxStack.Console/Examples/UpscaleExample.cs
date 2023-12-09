using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.ImageUpscaler.Services;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxStack.Console.Runner
{
    public sealed class UpscaleExample : IExampleRunner
    {
        private readonly string _outputDirectory;
        private readonly IOnnxModelService _modelService;
        private readonly IUpscaleService _imageUpscaleService;
       

        public UpscaleExample(IOnnxModelService modelService, IUpscaleService imageUpscaleService)
        {
            _modelService = modelService;
            _imageUpscaleService = imageUpscaleService;
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", nameof(UpscaleExample));
            Directory.CreateDirectory(_outputDirectory);
        }

        public string Name => "Image Upscale Demo";

        public string Description => "Upscales images";

        public async Task RunAsync()
        {

            var modelset = new OnnxModelSetConfig
            {
                Name = "Upscaler",
                IsEnabled = true,
                ExecutionProvider = ExecutionProvider.DirectML,
                ModelConfigurations = new List<OnnxModelSessionConfig>
                {
                    new OnnxModelSessionConfig
                    {
                        Type = OnnxModelType.Upscaler,
                        OnnxModelPath = "D:\\Repositories\\upscaler\\SwinIR\\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx"
                    }
                }
            };
            modelset.ApplyConfigurationOverrides();
            _modelService.UpdateModelSet(modelset);

            OutputHelpers.WriteConsole("Enter Image Path", ConsoleColor.Yellow);
            var imageFile = OutputHelpers.ReadConsole(ConsoleColor.Gray);
            if (!File.Exists(imageFile))
            {
                OutputHelpers.WriteConsole("File not found!", ConsoleColor.Red);
                return;
            }

            OutputHelpers.WriteConsole("Loading Model...", ConsoleColor.Cyan);
            await _modelService.LoadModelAsync(modelset);
            OutputHelpers.WriteConsole("Model Loaded.", ConsoleColor.Cyan);

            var inputImage = await Image.LoadAsync<Rgba32>(imageFile);

            OutputHelpers.WriteConsole("Upscaling Image...", ConsoleColor.Cyan);
            var result = await _imageUpscaleService.GenerateAsync(modelset, inputImage);
            await result.SaveAsPngAsync(Path.Combine(_outputDirectory, "Result.png"));
            OutputHelpers.WriteConsole("Upscaling Complete.", ConsoleColor.Cyan);
        }

    }
}

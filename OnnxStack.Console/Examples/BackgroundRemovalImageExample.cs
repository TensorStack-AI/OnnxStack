using OnnxStack.Core.Image;
using OnnxStack.FeatureExtractor.Pipelines;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class BackgroundRemovalImageExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public BackgroundRemovalImageExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", "BackgroundRemovalExample");
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 20;

        public string Name => "Image Background Removal Example";

        public string Description => "Remove a background from an image";

        /// <summary>
        /// ControlNet Example
        /// </summary>
        public async Task RunAsync()
        {
            OutputHelpers.WriteConsole("Please enter an image file path and press ENTER", ConsoleColor.Yellow);
            var imageFile = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

            var timestamp = Stopwatch.GetTimestamp();

            OutputHelpers.WriteConsole($"Load Image", ConsoleColor.Gray);
            var inputImage = await OnnxImage.FromFileAsync(imageFile);

            OutputHelpers.WriteConsole($"Create Pipeline", ConsoleColor.Gray);
            var pipeline = BackgroundRemovalPipeline.CreatePipeline("D:\\Repositories\\RMBG-1.4\\onnx\\model.onnx", sampleSize: 1024);

            OutputHelpers.WriteConsole($"Run Pipeline", ConsoleColor.Gray);
            var imageFeature = await pipeline.RunAsync(inputImage);

            OutputHelpers.WriteConsole($"Save Image", ConsoleColor.Gray);
            await imageFeature.SaveAsync(Path.Combine(_outputDirectory, $"{pipeline.Name}.png"));

            OutputHelpers.WriteConsole($"Unload pipeline", ConsoleColor.Gray);
            await pipeline.UnloadAsync();

            OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);
        }
    }
}

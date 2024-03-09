using OnnxStack.Core.Video;
using OnnxStack.FeatureExtractor.Pipelines;
using System.Diagnostics;

namespace OnnxStack.Console.Runner
{
    public sealed class BackgroundRemovalVideoExample : IExampleRunner
    {
        private readonly string _outputDirectory;

        public BackgroundRemovalVideoExample()
        {
            _outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "Examples", "BackgroundRemovalExample");
            Directory.CreateDirectory(_outputDirectory);
        }

        public int Index => 21;

        public string Name => "Video Background Removal Example";

        public string Description => "Remove a background from an video";

        public async Task RunAsync()
        {
            OutputHelpers.WriteConsole("Please enter an video/gif file path and press ENTER", ConsoleColor.Yellow);
            var videoFile = OutputHelpers.ReadConsole(ConsoleColor.Cyan);

            var timestamp = Stopwatch.GetTimestamp();

            OutputHelpers.WriteConsole($"Read Video", ConsoleColor.Gray);
            var videoInfo = await VideoHelper.ReadVideoInfoAsync(videoFile);

            OutputHelpers.WriteConsole($"Create Pipeline", ConsoleColor.Gray);
            var pipeline = BackgroundRemovalPipeline.CreatePipeline("D:\\Repositories\\RMBG-1.4\\onnx\\model.onnx", sampleSize: 1024);

            OutputHelpers.WriteConsole($"Load Pipeline", ConsoleColor.Gray);
            await pipeline.LoadAsync();

            OutputHelpers.WriteConsole($"Create Video Stream", ConsoleColor.Gray);
            var videoStream = VideoHelper.ReadVideoStreamAsync(videoFile, videoInfo.FrameRate);

            OutputHelpers.WriteConsole($"Create Pipeline Stream", ConsoleColor.Gray);
            var pipelineStream = pipeline.RunAsync(videoStream);

            OutputHelpers.WriteConsole($"Write Video Stream", ConsoleColor.Gray);
            await VideoHelper.WriteVideoStreamAsync(videoInfo, pipelineStream, Path.Combine(_outputDirectory, $"Result.mp4"), true);

            OutputHelpers.WriteConsole($"Unload", ConsoleColor.Gray);
            await pipeline.UnloadAsync();

            OutputHelpers.WriteConsole($"Elapsed: {Stopwatch.GetElapsedTime(timestamp)}ms", ConsoleColor.Yellow);
        }
    }
}

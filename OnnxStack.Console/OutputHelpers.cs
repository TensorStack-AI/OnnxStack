using OnnxStack.StableDiffusion.Common;

namespace OnnxStack.Console
{
    internal static class OutputHelpers
    {
        public static string ReadConsole(ConsoleColor color)
        {
            var previous = System.Console.ForegroundColor;
            System.Console.ForegroundColor = color;
            var line = System.Console.ReadLine();
            System.Console.ForegroundColor = previous;
            return line;
        }


        public static void WriteConsole(string value, ConsoleColor color, bool line = true)
        {
            var previous = System.Console.ForegroundColor;
            System.Console.ForegroundColor = color;
            if (line)
                System.Console.WriteLine(value);
            else
                System.Console.Write(value);
            System.Console.ForegroundColor = previous;
        }

        public static IProgress<DiffusionProgress> ProgressCallback => new Progress<DiffusionProgress>((DiffusionProgress progress) => WriteConsole(PrintProgress(progress), ConsoleColor.Gray));
        public static IProgress<DiffusionProgress> BatchProgressCallback => new Progress<DiffusionProgress>((DiffusionProgress progress) => WriteConsole(PrintBatchProgress(progress), ConsoleColor.Gray));
        public static IProgress<DiffusionProgress> FrameProgressCallback => new Progress<DiffusionProgress>((DiffusionProgress progress) => WriteConsole(PrintFrameProgress(progress), ConsoleColor.Gray));


        private static string PrintProgress(DiffusionProgress progress)
        {
            if (progress.StepMax == 0)
                return progress.Elapsed > 0 ? $"Complete: {progress.Elapsed}ms" : progress.Message;

            return progress.Elapsed > 0 ? $"Step: {progress.StepValue}/{progress.StepMax} - {progress.Elapsed}ms" : progress.Message;
        }

        private static string PrintBatchProgress(DiffusionProgress progress)
        {
            if (progress.StepMax == 0)
                return progress.Elapsed > 0 ? $"Complete: {progress.Elapsed}ms" : progress.Message;

            return $"Batch: {progress.BatchValue}/{progress.BatchMax} - Step: {progress.StepValue}/{progress.StepMax} - {progress.Elapsed}ms";
        }

        private static string PrintFrameProgress(DiffusionProgress progress)
        {
            if (progress.StepMax == 0)
                return progress.Elapsed > 0 ? $"Complete: {progress.Elapsed}ms" : progress.Message;

            return $"Frame: {progress.BatchValue}/{progress.BatchMax} - Step: {progress.StepValue}/{progress.StepMax} - {progress.Elapsed}ms";
        }


        public static IProgress<DiffusionProgress> PythonProgressCallback => new Progress<DiffusionProgress>((DiffusionProgress progress) => WriteConsole(PrintPythonProgress(progress), ConsoleColor.Gray));

        private static string PrintPythonProgress(DiffusionProgress progress)
        {
            var msg = string.IsNullOrEmpty(progress.Message) ? string.Empty : $"{progress.Message.PadRight(30)}\t" ;
            return $"[PythonProgress] {msg} Step: {progress.StepValue}/{progress.StepMax}\t\t{progress.IterationsPerSecond:F2} it/s\t{progress.SecondsPerIteration:F2} s/it\t\t{progress.Downloaded}M/{progress.DownloadTotal}M @ {progress.DownloadSpeed}MB/s\t\t{progress.Percentage}%";
        }
    }
}

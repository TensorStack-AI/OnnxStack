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

        public static Action<DiffusionProgress> ProgressCallback => (DiffusionProgress progress) => WriteConsole($"Step: {progress.StepValue}/{progress.StepMax}", ConsoleColor.Gray);
        public static Action<DiffusionProgress> BatchProgressCallback => (DiffusionProgress progress) => WriteConsole($"Batch: {progress.BatchValue}/{progress.BatchMax} - Step: {progress.StepValue}/{progress.StepMax}", ConsoleColor.Gray);
        public static Action<DiffusionProgress> FrameProgressCallback => (DiffusionProgress progress) => WriteConsole($"Frame: {progress.BatchValue}/{progress.BatchMax} - Step: {progress.StepValue}/{progress.StepMax}", ConsoleColor.Gray);
    }
}

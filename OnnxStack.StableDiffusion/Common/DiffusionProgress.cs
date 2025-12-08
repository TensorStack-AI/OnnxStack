using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

namespace OnnxStack.StableDiffusion.Common
{
    public record DiffusionProgress
    {
        public DiffusionProgress() { }
        public DiffusionProgress(string message)
        {
            Message = message;
        }
        public DiffusionProgress(long elapsed)
        {
            Elapsed = Stopwatch.GetElapsedTime(elapsed).TotalMilliseconds;
        }
        public string Message { get; set; }

        public int BatchMax { get; set; }
        public int BatchValue { get; set; }
        public DenseTensor<float> BatchTensor { get; set; }

        public int StepMax { get; set; }
        public int StepValue { get; set; }
        public DenseTensor<float> StepTensor { get; set; }

        public double Elapsed { get; set; }



        public int Percentage { get; set; }
        public float IterationsPerSecond { get; set; }
        public float SecondsPerIteration { get; set; }
        public float Downloaded { get; set; }
        public float DownloadTotal { get; set; }
        public float DownloadSpeed { get; set; }
    }
}

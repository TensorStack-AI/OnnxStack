using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using System;
using System.Text.Json.Serialization;
using System.Windows.Media.Imaging;

namespace OnnxStack.UI.Models
{
    public class ImageResult
    {
        [JsonIgnore]
        public BitmapSource Image { get; init; }

        public DateTime Timestamp { get; } = DateTime.UtcNow;
        public ProcessType ProcessType { get; init; }
        public string Prompt { get; init; }
        public string NegativePrompt { get; init; }
        public SchedulerType SchedulerType { get; init; }
        public SchedulerOptions SchedulerOptions { get; init; }
        public double Elapsed { get; init; }
    }
}
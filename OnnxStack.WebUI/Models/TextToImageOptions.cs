using OnnxStack.StableDiffusion.Enums;
using System.ComponentModel.DataAnnotations;

namespace OnnxStack.WebUI.Models
{
    public class TextToImageOptions
    {
        [Required]
        [StringLength(512, MinimumLength = 2)]
        public string Prompt { get; set; }

        [StringLength(512)]
        public string NegativePrompt { get; set; }
        public SchedulerType SchedulerType { get; set; }

        [Range(64, 1024)]
        public int Width { get; set; } = 512;

        [Range(64, 1024)]
        public int Height { get; set; } = 512;

        [Range(0, int.MaxValue)]
        public int Seed { get; set; }

        [Range(1, 100)]
        public int InferenceSteps { get; set; } = 30;

        [Range(0f, 40f)]
        public float GuidanceScale { get; set; } = 7.5f;

        [Range(0f, 1f)]
        public float Strength { get; set; } = 0.6f;

        [Range(-1f, 1f)]
        public float InitialNoiseLevel { get; set; } = 0f;
    }
}

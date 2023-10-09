using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace OnnxStack.StableDiffusion.Config
{
    public class PromptOptions
    {
        [Required]
        [StringLength(512, MinimumLength = 4)]
        public string Prompt { get; set; }

        [StringLength(512)]
        public string NegativePrompt { get; set; }
        public SchedulerType SchedulerType { get; set; }

        public InputImage InputImage { get; set; }

        public InputImage InputImageMask { get; set; }

        public bool HasInputImage => InputImage?.HasImage ?? false;
        public bool HasInputImageMask => InputImageMask?.HasImage ?? false;
    }
}

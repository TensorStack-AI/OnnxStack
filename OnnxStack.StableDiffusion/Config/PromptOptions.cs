using OnnxStack.StableDiffusion.Enums;
using System.ComponentModel.DataAnnotations;

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
        public string InputImage { get; set; }
        public bool HasInputImage => !string.IsNullOrEmpty(InputImage);
    }
}

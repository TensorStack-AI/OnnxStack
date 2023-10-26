using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System.ComponentModel.DataAnnotations;

namespace OnnxStack.StableDiffusion.Config
{
    public class PromptOptions
    {
        public DiffuserType DiffuserType { get; set; }

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

    public enum DiffuserType
    {
        TextToImage = 0,
        ImageToImage = 1,
        ImageInpaint = 2,
        ImageInpaintLegacy = 3
    }
}

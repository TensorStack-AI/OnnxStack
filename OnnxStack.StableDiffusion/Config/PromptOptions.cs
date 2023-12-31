using OnnxStack.Core.Image;
using OnnxStack.Core.Video;
using OnnxStack.StableDiffusion.Enums;
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

        public InputImage InputImage { get; set; }

        public InputImage InputImageMask { get; set; }

        public VideoFrames InputVideo { get; set; }

        public bool HasInputVideo => InputVideo?.Frames?.Count > 0;
        public bool HasInputImage => InputImage?.HasImage ?? false;
        public bool HasInputImageMask => InputImageMask?.HasImage ?? false;
    }
}

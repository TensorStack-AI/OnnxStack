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

        public OnnxImage InputImage { get; set; }

        public OnnxImage InputImageMask { get; set; }

        public OnnxImage InputContolImage { get; set; }

        public OnnxVideo InputVideo { get; set; }
        public OnnxVideo InputContolVideo { get; set; }

        public bool HasInputVideo => InputVideo?.HasVideo ?? false;
        public bool HasInputImage => InputImage?.HasImage ?? false;
        public bool HasInputImageMask => InputImageMask?.HasImage ?? false;
    }
}

using OnnxStack.StableDiffusion.Enums;

namespace OnnxStack.StableDiffusion.Config
{
    public record BatchOptions
    {
        public BatchOptionType BatchType { get; set; }
        public float ValueTo { get; set; }
        public float ValueFrom { get; set; }
        public float Increment { get; set; } = 1f;
    }
}

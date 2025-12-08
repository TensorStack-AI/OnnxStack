using OnnxStack.Core.Config;

namespace OnnxStack.StableDiffusion.Tokenizers
{
    public record TokenizerConfig : OnnxModelConfig
    {
        public int TokenizerLimit { get; set; } = 77;
        public int TokenizerLength { get; set; } = 768;
        public int PadTokenId { get; set; } = 49407;
        public int BlankTokenId { get; set; } = 49407;
    }
}

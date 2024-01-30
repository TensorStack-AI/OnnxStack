using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class TokenizerModel : OnnxModelSession
    {
        private readonly int _padTokenId;
        private readonly int _blankTokenId;
        private readonly int _tokenizerLimit;
        private readonly int _tokenizerLength;

        public TokenizerModel(TokenizerModelConfig configuration) : base(configuration)
        {
            _padTokenId = configuration.PadTokenId;
            _blankTokenId = configuration.BlankTokenId;
            _tokenizerLimit = configuration.TokenizerLimit;
            _tokenizerLength = configuration.TokenizerLength;
        }

        public int TokenizerLimit => _tokenizerLimit;
        public int TokenizerLength => _tokenizerLength;
        public int PadTokenId => _padTokenId;
        public int BlankTokenId => _blankTokenId;
    }

    public record TokenizerModelConfig : OnnxModelConfig
    {
        public int TokenizerLimit { get; set; } = 77;
        public int TokenizerLength { get; set; } = 768;
        public int PadTokenId { get; set; } = 49407;
        public int BlankTokenId { get; set; } = 49407;
    }
}

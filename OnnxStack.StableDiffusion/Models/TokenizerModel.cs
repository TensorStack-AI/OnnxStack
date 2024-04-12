using OnnxStack.Core.Config;
using OnnxStack.Core.Model;

namespace OnnxStack.StableDiffusion.Models
{
    public class TokenizerModel : OnnxModelSession
    {
        private readonly TokenizerModelConfig _configuration;

        public TokenizerModel(TokenizerModelConfig configuration) : base(configuration)
        {
            _configuration = configuration;
        }

        public int TokenizerLimit => _configuration.TokenizerLimit;
        public int TokenizerLength => _configuration.TokenizerLength;
        public int PadTokenId => _configuration.PadTokenId;
        public int BlankTokenId => _configuration.BlankTokenId;
    }

    public record TokenizerModelConfig : OnnxModelConfig
    {
        public int TokenizerLimit { get; set; } = 77;
        public int TokenizerLength { get; set; } = 768;
        public int PadTokenId { get; set; } = 49407;
        public int BlankTokenId { get; set; } = 49407;
    }
}

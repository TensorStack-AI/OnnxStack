using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Config;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Enums;

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

        public static TokenizerModel Create(TokenizerModelConfig configuration)
        {
            return new TokenizerModel(configuration);
        }

        public static TokenizerModel Create(string modelFile, int tokenizerLength = 768, int tokenizerLimit = 77, int padTokenId = 49407, int blankTokenId = 49407, int deviceId = 0, ExecutionProvider executionProvider = ExecutionProvider.DirectML)
        {
            var configuration = new TokenizerModelConfig
            {
                DeviceId = deviceId,
                ExecutionProvider = executionProvider,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                InterOpNumThreads = 0,
                IntraOpNumThreads = 0,
                OnnxModelPath = modelFile,
                PadTokenId = padTokenId,
                BlankTokenId = blankTokenId,
                TokenizerLength = tokenizerLength,
                TokenizerLimit = tokenizerLimit
            };
            return new TokenizerModel(configuration);
        }
    }

    public record TokenizerModelConfig : OnnxModelConfig
    {
        public int TokenizerLimit { get; set; } = 77;
        public int TokenizerLength { get; set; } = 768;
        public int PadTokenId { get; set; } = 49407;
        public int BlankTokenId { get; set; } = 49407;
    }
}

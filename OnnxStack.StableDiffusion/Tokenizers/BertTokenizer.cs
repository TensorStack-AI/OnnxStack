using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Models;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Tokenizers
{
    public class BertTokenizer : ITokenizer
    {
        private readonly int _bos = 101;
        private readonly int _eos = 102;
        private readonly int[] _specialTokens;
        private readonly TokenizerConfig _configuration;
        private Microsoft.ML.Tokenizers.BertTokenizer _bertTokenizer;

        public BertTokenizer(TokenizerConfig configuration)
        {
            _configuration = configuration;
            _specialTokens = [_bos, _eos];
            var directory = Path.GetDirectoryName(configuration.OnnxModelPath);
            var vocabFile = Path.Combine(directory, "vocab.txt");
            _bertTokenizer = Microsoft.ML.Tokenizers.BertTokenizer.Create(File.OpenRead(vocabFile));
        }

        public int TokenizerLength => _configuration.TokenizerLength;
        public int TokenizerLimit => _configuration.TokenizerLimit;
        public int PadTokenId => _configuration.PadTokenId;
        public int BlankTokenId => _configuration.BlankTokenId;

        public Task<TokenizerResult> EncodeAsync(string text)
        {
            var inputIds = _bertTokenizer.EncodeToIds(text).ToArray().ToLong();
            var attentionMask = Enumerable.Repeat<long>(1, inputIds.Length).ToArray();
            return Task.FromResult(new TokenizerResult(inputIds, attentionMask));
        }

        public void Dispose()
        {
            _bertTokenizer = null;
        }
    }
}

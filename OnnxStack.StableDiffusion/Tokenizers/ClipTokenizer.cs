using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Models;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Tokenizers
{
    public class ClipTokenizer : ITokenizer
    {
        private readonly int _bos = 49406;
        private readonly int _eos = 49407;
        private readonly int[] _specialTokens;
        private readonly TokenizerConfig _configuration;
        private Microsoft.ML.Tokenizers.BpeTokenizer _bpeTokenizer;

        public int TokenizerLength => _configuration.TokenizerLength;
        public int TokenizerLimit => _configuration.TokenizerLimit;
        public int PadTokenId => _configuration.PadTokenId;
        public int BlankTokenId => _configuration.BlankTokenId;

        public ClipTokenizer(TokenizerConfig configuration)
        {
            _configuration = configuration;
            _specialTokens = [_bos, _eos];
            var directory = Path.GetDirectoryName(configuration.OnnxModelPath);
            var vocabFile = Path.Combine(directory, "vocab.json");
            var mergesFile = Path.Combine(directory, "merges.txt");
            _bpeTokenizer = Microsoft.ML.Tokenizers.BpeTokenizer.Create(vocabFile, mergesFile, normalizer: new Microsoft.ML.Tokenizers.LowerCaseNormalizer(), unknownToken: "<|endoftext|>", endOfWordSuffix: "</w>");
        }

        public Task<TokenizerResult> EncodeAsync(string text)
        {
            var result = _bpeTokenizer.EncodeToIds(text).ToArray().ToLong();
            var inputIds = new List<long>([_bos]);
            inputIds.AddRange(result);
            inputIds.Add(_eos);
            var attentionMask = Enumerable.Repeat<long>(1, inputIds.Count).ToArray();
            return Task.FromResult(new TokenizerResult(inputIds.ToArray(), attentionMask));
        }

        public Task<string> DecodeAsync(long[] tokens)
        {
            var result = _bpeTokenizer.Decode(tokens.ToInt(), false);
            return Task.FromResult(result);
        }

        public void Dispose()
        {
            _bpeTokenizer = null;
        }
    }
}

using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Tokenizers
{
    public class SentencePieceTokenizer : ITokenizer
    {
        private readonly TokenizerConfig _configuration;
        private Microsoft.ML.Tokenizers.SentencePieceTokenizer _sentencePieceTokenizer;

        public SentencePieceTokenizer(TokenizerConfig configuration)
        {
            _configuration = configuration;
            _sentencePieceTokenizer = CreateTokenizer();
        }

        public int TokenizerLength => _configuration.TokenizerLength;
        public int TokenizerLimit => _configuration.TokenizerLimit;
        public int PadTokenId => _configuration.PadTokenId;
        public int BlankTokenId => _configuration.BlankTokenId;
 

        public Task<TokenizerResult> EncodeAsync(string text)
        {
            var inputIds = _sentencePieceTokenizer.EncodeToIds(text).ToArray().ToLong();
            var attentionMask = Enumerable.Repeat<long>(1, inputIds.Length).ToArray();
            return Task.FromResult(new TokenizerResult(inputIds, attentionMask));
        }

        public void Dispose()
        {
            _sentencePieceTokenizer = null;
        }


        private Microsoft.ML.Tokenizers.SentencePieceTokenizer CreateTokenizer()
        {
            var specialTokens = GetSpecialTokens(_configuration.OnnxModelPath);
            using (var fileStream = File.OpenRead(_configuration.OnnxModelPath))
            {
                return Microsoft.ML.Tokenizers.SentencePieceTokenizer.Create(fileStream, addBeginOfSentence: false, addEndOfSentence: true, specialTokens);
            }
        }


        private Dictionary<string, int> GetSpecialTokens(string tokeizerModelPath)
        {
            try
            {
                var tokenizerConfig = Path.Combine(Path.GetDirectoryName(tokeizerModelPath), "tokenizer.json");
                if (!File.Exists(tokenizerConfig))
                    return null;

                using (var tokenizerConfigFile = File.OpenRead(tokenizerConfig))
                {
                    var sentencePieceConfig = JsonSerializer.Deserialize<SentencePieceConfig>(tokenizerConfigFile);
                    if (sentencePieceConfig is null || sentencePieceConfig.AddedTokens is null)
                        return null;


                    return sentencePieceConfig.AddedTokens.ToDictionary(k => k.Content, v => v.Id);
                }
            }
            catch (Exception)
            {
                return null;
            }
        }


        private record SentencePieceConfig
        {
            [JsonPropertyName("added_tokens")]
            public AddedToken[] AddedTokens { get; set; }
        }


        private record AddedToken
        {
            [JsonPropertyName("id")]
            public int Id { get; set; }

            [JsonPropertyName("content")]
            public string Content { get; set; }
        }
    }
}

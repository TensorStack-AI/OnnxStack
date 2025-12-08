using OnnxStack.StableDiffusion.Common;
using System;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Tokenizers
{
    public interface ITokenizer : IDisposable
    {
        int TokenizerLength { get; }
        int TokenizerLimit { get; }
        int PadTokenId { get; }
        int BlankTokenId { get; }
        Task<TokenizerResult> EncodeAsync(string text);
    }
}
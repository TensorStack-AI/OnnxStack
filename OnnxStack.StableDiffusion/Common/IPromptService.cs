using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IPromptService
    {
        Task<DenseTensor<float>> CreatePromptAsync(string prompt, string negativePrompt);
        Task<int[]> DecodeTextAsync(string inputText);
        Task<float[]> EncodeTokensAsync(int[] tokenizedInput);
    }
}
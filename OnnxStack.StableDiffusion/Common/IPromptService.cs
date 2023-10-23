using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IPromptService
    {
        Task<DenseTensor<float>> CreatePromptAsync(IModelOptions model, string prompt, string negativePrompt);
        Task<int[]> DecodeTextAsync(IModelOptions model, string inputText);
        Task<float[]> EncodeTokensAsync(IModelOptions model, int[] tokenizedInput);
    }
}
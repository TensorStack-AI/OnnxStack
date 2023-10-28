using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IPromptService
    {
        Task<DenseTensor<float>> CreatePromptAsync(IModelOptions model, PromptOptions promptOptions, SchedulerOptions schedulerOptions);
        Task<int[]> DecodeTextAsync(IModelOptions model, string inputText);
        Task<float[]> EncodeTokensAsync(IModelOptions model, int[] tokenizedInput);
    }
}
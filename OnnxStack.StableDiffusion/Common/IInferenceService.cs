using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IInferenceService
    {
        Task<int[]> GetTokens(string text);
        Task<DenseTensor<float>> RunInference(StableDiffusionOptions options, SchedulerOptions schedulerOptions);
    }
}
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IInferenceService
    {
        Task<int[]> TokenizeAsync(string text);
        Task<DenseTensor<float>> RunInferenceAsync(StableDiffusionOptions options, SchedulerOptions schedulerOptions);
    }
}
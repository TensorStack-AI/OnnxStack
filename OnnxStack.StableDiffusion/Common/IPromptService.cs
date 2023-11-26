using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.StableDiffusion.Config;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IPromptService
    {
        Task<PromptEmbeddingsResult> CreatePromptAsync(IModelOptions model, PromptOptions promptOptions, bool isGuidanceEnabled);
    }
}
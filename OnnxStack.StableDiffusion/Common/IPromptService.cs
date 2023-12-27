using OnnxStack.StableDiffusion.Config;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Common
{
    public interface IPromptService
    {
        Task<PromptEmbeddingsResult> CreatePromptAsync(StableDiffusionModelSet model, PromptOptions promptOptions, bool isGuidanceEnabled);
    }
}
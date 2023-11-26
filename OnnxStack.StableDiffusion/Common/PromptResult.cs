using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxStack.StableDiffusion.Common
{
    public record PromptEmbeddingsResult(DenseTensor<float> PromptEmbeds, DenseTensor<float> PooledPromptEmbeds = default);
}

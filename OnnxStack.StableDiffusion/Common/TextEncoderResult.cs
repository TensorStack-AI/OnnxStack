using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxStack.StableDiffusion.Common
{
    public record TextEncoderResult(DenseTensor<float> PromptEmbeds, DenseTensor<float> PooledPromptEmbeds)
    {
    }
}

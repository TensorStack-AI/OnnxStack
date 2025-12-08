using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxStack.StableDiffusion.Common
{
    public record EncoderResult(DenseTensor<float> PromptEmbeds, DenseTensor<float> PooledPromptEmbeds);
}

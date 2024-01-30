using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxStack.StableDiffusion.Common
{
    public record PromptEmbeddingsResult(DenseTensor<float> PromptEmbeds, DenseTensor<float> PooledPromptEmbeds = default);

    public record EncoderResult(float[] PromptEmbeds, float[] PooledPromptEmbeds);

    public record EmbedsResult(DenseTensor<float> PromptEmbeds, DenseTensor<float> PooledPromptEmbeds);
}

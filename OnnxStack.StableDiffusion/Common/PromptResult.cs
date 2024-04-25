using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxStack.StableDiffusion.Common
{
    public record PromptEmbeddingsResult(DenseTensor<float> PromptEmbeds, DenseTensor<float> PooledPromptEmbeds = default);

    public record EncoderResult(DenseTensor<float> PromptEmbeds, DenseTensor<float> PooledPromptEmbeds);

    public record TokenizerResult
    {
        public TokenizerResult(long[] inputIds, long[] attentionMask)
        {
            InputIds = inputIds;
            AttentionMask = attentionMask;
        }

        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
    }
}

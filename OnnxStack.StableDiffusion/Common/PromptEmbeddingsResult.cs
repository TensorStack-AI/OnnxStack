using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;

namespace OnnxStack.StableDiffusion.Common
{
    public record PromptEmbeddingsResult
    {
        private readonly DenseTensor<float> _promptEmbeds;
        private readonly DenseTensor<float> _pooledPromptEmbeds;
        private readonly DenseTensor<float> _negativePromptEmbeds;
        private readonly DenseTensor<float> _negativePooledPromptEmbeds;

        public PromptEmbeddingsResult(DenseTensor<float> promptEmbeds, DenseTensor<float> pooledPromptEmbeds, DenseTensor<float> negativePromptEmbeds, DenseTensor<float> negativePooledPromptEmbeds)
        {
            _promptEmbeds = promptEmbeds;
            _pooledPromptEmbeds = pooledPromptEmbeds;
            _negativePromptEmbeds = negativePromptEmbeds;
            _negativePooledPromptEmbeds = negativePooledPromptEmbeds;
        }


        public DenseTensor<float> PromptEmbeds => _promptEmbeds;
        public DenseTensor<float> PooledPromptEmbeds => _pooledPromptEmbeds;
        public DenseTensor<float> NegativePromptEmbeds => _negativePromptEmbeds;
        public DenseTensor<float> NegativePooledPromptEmbeds => _negativePooledPromptEmbeds;

        public DenseTensor<float> GetPromptEmbeds(bool classifierFreeGuidance)
        {
            if (classifierFreeGuidance)
                return _negativePromptEmbeds.Concatenate(_promptEmbeds);

            return _promptEmbeds;
        }

        public DenseTensor<float> GetPooledPromptEmbeds(bool classifierFreeGuidance)
        {
            if (classifierFreeGuidance)
                return _negativePooledPromptEmbeds.Concatenate(_pooledPromptEmbeds);

            return _pooledPromptEmbeds;
        }
    }
}

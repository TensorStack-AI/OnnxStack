using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Schedulers.LatentConsistency;
using System;

namespace OnnxStack.StableDiffusion.Schedulers
{
    public sealed class LocomotionScheduler : LCMScheduler
    {
        public LocomotionScheduler(SchedulerOptions options)
            : base(options) { }

        protected override DenseTensor<float> CreateRandomSample(ReadOnlySpan<int> dimensions, int contextSize)
        {
            var batch = dimensions[0];
            var frames = dimensions[2];
            var channels = dimensions[1];
            var height = dimensions[3];
            var width = dimensions[4];
            if (frames == contextSize)
                return CreateRandomSample(dimensions);

            int[] contextDimensions = [batch, contextSize, channels, height, width];
            return CreateRandomSample(contextDimensions)
                .Repeat(frames / contextSize)
                .ReshapeTensor([1, frames, channels, height, width])
                .Permute([0, 2, 1, 3, 4]); //BCFHW
        }
    }
}

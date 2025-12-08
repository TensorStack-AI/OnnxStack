using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using System;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    public sealed class FlowMatchEulerDynamicScheduler : FlowMatchEulerDiscreteScheduler
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FlowMatchEulerDynamicScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public FlowMatchEulerDynamicScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="FlowMatchEulerDynamicScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public FlowMatchEulerDynamicScheduler(SchedulerOptions schedulerOptions) : base(schedulerOptions) { }


        /// <summary>
        /// Processes a inference step for the specified model output.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="order">The order.</param>
        /// <returns></returns>
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int contextSize = 16)
        {
            var stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];
            var sigmaNext = Sigmas[stepIndex + 1];

            var noise = CreateRandomSample(modelOutput.Dimensions);
            var prevSample = noise
                .MultiplyTensorByFloat(sigmaNext)
                .AddTensors(sample
                    .Subtract(modelOutput.MultiplyTensorByFloat(sigma))
                    .MultiplyTensorByFloat(1f - sigmaNext));
            return new SchedulerStepResult(prevSample);
        }

    }
}

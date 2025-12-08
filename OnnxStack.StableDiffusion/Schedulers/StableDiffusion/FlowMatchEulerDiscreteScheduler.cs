using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    public class FlowMatchEulerDiscreteScheduler : SchedulerBase
    {
        private float _sigmaMin;
        private float _sigmaMax;

        /// <summary>
        /// Initializes a new instance of the <see cref="FlowMatchEulerDiscreteScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public FlowMatchEulerDiscreteScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="FlowMatchEulerDiscreteScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public FlowMatchEulerDiscreteScheduler(SchedulerOptions schedulerOptions) : base(schedulerOptions) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            base.Initialize();
            var timesteps = ArrayHelpers.Linspace(1, Options.TrainTimesteps, Options.TrainTimesteps);
            var sigmas = timesteps
                .Select(x => x / Options.TrainTimesteps)
                .Select(sigma => Options.Shift * sigma / (1f + (Options.Shift - 1f) * sigma))
                .ToArray();
            _sigmaMin = sigmas.Min();
            _sigmaMax = sigmas.Max();
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            var timesteps = ArrayHelpers.Linspace(SigmaToTimestep(_sigmaMin), SigmaToTimestep(_sigmaMax), Options.InferenceSteps);
            if (Options.InferenceSteps == 1)
                timesteps = [Options.TrainTimesteps];

            var sigmas = timesteps
                .Select(x => x / Options.TrainTimesteps)
                .Select(sigma => Options.Shift * sigma / (1f + (Options.Shift - 1f) * sigma))
                .Reverse();

            Sigmas = [.. sigmas, 0f];

            var timestepValues = sigmas
                 .Select(sigma => sigma * Options.TrainTimesteps)
                 .Select(x => (int)Math.Round(x))
                 .OrderByDescending(x => x)
                 .ToArray();
            return timestepValues;
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="sample">The sample.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public override DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            return sample;
        }


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

            var prevSample = modelOutput
                .MultiplyTensorByFloat(sigmaNext - sigma)
                .AddTensors(sample);
            return new SchedulerStepResult(prevSample);
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="originalSamples">The original samples.</param>
        /// <param name="noise">The noise.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        public override DenseTensor<float> AddNoise(DenseTensor<float> originalSamples, DenseTensor<float> noise, IReadOnlyList<int> timesteps)
        {
            var sigma = timesteps
                .Select(x => Timesteps.IndexOf(x))
                .Select(x => Sigmas[x])
                .Max();

            return noise
                .MultiplyTensorByFloat(sigma)
                .AddTensors(originalSamples.MultiplyBy(1f - sigma));
        }


        /// <summary>
        /// Sigmas to timestep.
        /// </summary>
        /// <param name="sigma">The sigma.</param>
        /// <returns>System.Single.</returns>
        private float SigmaToTimestep(float sigma)
        {
            return sigma * Options.TrainTimesteps;
        }
    }
}

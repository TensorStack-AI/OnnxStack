using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    public sealed class EulerAncestralScheduler : SchedulerBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EulerAncestralScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public EulerAncestralScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="EulerAncestralScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public EulerAncestralScheduler(SchedulerOptions schedulerOptions) : base(schedulerOptions) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            base.Initialize();
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            var sigmas = Sigmas.ToArray();
            var timesteps = GetTimesteps();
            var logSigmas = ArrayHelpers.Log(sigmas);
            var range = ArrayHelpers.Range(0, sigmas.Length, true);
            sigmas = Interpolate(timesteps, range, sigmas);

            if (Options.UseKarrasSigmas)
            {
                sigmas = ConvertToKarras(sigmas);
                timesteps = SigmaToTimestep(sigmas, logSigmas);
            }

            Sigmas = [.. sigmas, 0f];

            SetInitNoiseSigma();

            return timesteps.Select(x => (int)Math.Round(x))
                 .OrderByDescending(x => x)
                 .ToArray();
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="sample">The sample.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public override DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            var stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];
            sigma = MathF.Sqrt(MathF.Pow(sigma, 2f) + 1f);
            return sample.DivideTensorByFloat(sigma);
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

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = GetPredictedSample(modelOutput, sample, sigma);

            var sigmaFrom = Sigmas[stepIndex];
            var sigmaTo = Sigmas[stepIndex + 1];

            var sigmaFromLessSigmaTo = MathF.Pow(sigmaFrom, 2) - MathF.Pow(sigmaTo, 2);
            var sigmaUpResult = MathF.Pow(sigmaTo, 2) * sigmaFromLessSigmaTo / MathF.Pow(sigmaFrom, 2);
            var sigmaUp = sigmaUpResult < 0 ? -MathF.Pow(MathF.Abs(sigmaUpResult), 0.5f) : MathF.Pow(sigmaUpResult, 0.5f);

            var sigmaDownResult = MathF.Pow(sigmaTo, 2) - MathF.Pow(sigmaUp, 2);
            var sigmaDown = sigmaDownResult < 0 ? -MathF.Pow(MathF.Abs(sigmaDownResult), 0.5f) : MathF.Pow(sigmaDownResult, 0.5f);

            // 2. Convert to an ODE derivative
            var derivative = sample
                .SubtractTensors(predOriginalSample)
                .DivideTensorByFloat(sigma);

            var delta = sigmaDown - sigma;
            var prevSample = sample.AddTensors(derivative.MultiplyTensorByFloat(delta));
            var noise = CreateRandomSample(prevSample.Dimensions);
            prevSample = prevSample.AddTensors(noise.MultiplyTensorByFloat(sigmaUp));
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
            // Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py#L389
            var sigma = timesteps
                .Select(x => Timesteps.IndexOf(x))
                .Select(x => Sigmas[x])
                .Max();

            return noise
                .MultiplyTensorByFloat(sigma)
                .AddTensors(originalSamples);
        }

    }
}

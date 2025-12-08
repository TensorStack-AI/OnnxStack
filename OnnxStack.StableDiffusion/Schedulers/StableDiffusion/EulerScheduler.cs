using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    public sealed class EulerScheduler : SchedulerBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EulerScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public EulerScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="EulerScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public EulerScheduler(SchedulerOptions schedulerOptions) : base(schedulerOptions) { }


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

            // TODO: Implement "interpolation_type"
            //var interpolation_type = "linear";
            //sigmas = interpolation_type == "log_linear"
            //    ? np.exp(np.linspace(np.log(sigmas.Last()), np.log(sigmas.First()), timesteps.Length + 1)).ToArray<float>()
            //    : Interpolate(timesteps, range, _sigmas);

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
            sigma = (float)Math.Sqrt(Math.Pow(sigma, 2) + 1);
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
            // TODO: Implement "extended settings for scheduler types"
            float s_churn = 0f;
            float s_tmin = 0f;
            float s_tmax = float.PositiveInfinity;
            float s_noise = 1f;

            var stepIndex = Timesteps.IndexOf(timestep);
            float sigma = Sigmas[stepIndex];

            float gamma = s_tmin <= sigma && sigma <= s_tmax ? (float)Math.Min(s_churn / (Sigmas.Length - 1f), Math.Sqrt(2.0f) - 1.0f) : 0f;
            var noise = CreateRandomSample(modelOutput.Dimensions);
            var epsilon = noise.MultiplyTensorByFloat(s_noise);
            float sigmaHat = sigma * (1.0f + gamma);

            if (gamma > 0)
                sample = sample.AddTensors(epsilon.MultiplyTensorByFloat((float)Math.Sqrt(Math.Pow(sigmaHat, 2f) - Math.Pow(sigma, 2f))));


            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = Options.PredictionType != PredictionType.Epsilon
                ? GetPredictedSample(modelOutput, sample, sigma)
                : sample.SubtractTensors(modelOutput.MultiplyTensorByFloat(sigmaHat));


            // 2. Convert to an ODE derivative
            var derivative = sample
                .SubtractTensors(predOriginalSample)
                .DivideTensorByFloat(sigmaHat);

            var delta = Sigmas[stepIndex + 1] - sigmaHat;
            return new SchedulerStepResult(sample.AddTensors(derivative.MultiplyTensorByFloat(delta)));
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

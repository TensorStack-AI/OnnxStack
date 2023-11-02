using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
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
        private float[] _sigmas;

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
            _sigmas = null;

            var betas = GetBetaSchedule();
            var alphas = betas.Select(beta => 1.0f - beta);
            var alphaCumProd = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b));
            _sigmas = alphaCumProd
                 .Select(alpha_prod => (float)Math.Sqrt((1 - alpha_prod) / alpha_prod))
                 .ToArray();

            var initNoiseSigma = GetInitNoiseSigma(_sigmas);
            SetInitNoiseSigma(initNoiseSigma);
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            var sigmas = _sigmas.ToArray();
            var timesteps = GetTimesteps();
            var log_sigmas = np.log(sigmas).ToArray<float>();
            var range = np.arange(0, (float)_sigmas.Length).ToArray<float>();

            // TODO: Implement "interpolation_type"
            var interpolation_type = "linear";
            sigmas = interpolation_type == "log_linear"
                ? np.exp(np.linspace(np.log(sigmas.Last()), np.log(sigmas.First()), timesteps.Length + 1)).ToArray<float>()
                : Interpolate(timesteps, range, _sigmas);

            if (Options.UseKarrasSigmas)
            {
                sigmas = ConvertToKarras(sigmas);
                timesteps = SigmaToTimestep(sigmas, log_sigmas);
            }

            _sigmas = sigmas
                .Append(0.000f)
                .ToArray();

            return timesteps.Select(x => (int)x)
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
            // Get step index of timestep from TimeSteps
            int stepIndex = Timesteps.IndexOf(timestep);

            // Get sigma at stepIndex
            var sigma = _sigmas[stepIndex];
            sigma = (float)Math.Sqrt(Math.Pow(sigma, 2) + 1);

            // Divide sample tensor shape {2,4,(H/8),(W/8)} by sigma
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
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            // TODO: Implement "extended settings for scheduler types"
            float s_churn = 0f;
            float s_tmin = 0f;
            float s_tmax = float.PositiveInfinity;
            float s_noise = 1f;

            var stepIndex = Timesteps.IndexOf(timestep);
            float sigma = _sigmas[stepIndex];

            float gamma = s_tmin <= sigma && sigma <= s_tmax ? (float)Math.Min(s_churn / (_sigmas.Length - 1f), Math.Sqrt(2.0f) - 1.0f) : 0f;
            var noise = CreateRandomSample(modelOutput.Dimensions);
            var epsilon = noise.MultipleTensorByFloat(s_noise);
            float sigmaHat = sigma * (1.0f + gamma);

            if (gamma > 0)
                sample = sample.AddTensors(epsilon.MultipleTensorByFloat((float)Math.Sqrt(Math.Pow(sigmaHat, 2f) - Math.Pow(sigma, 2f))));


            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = Options.PredictionType != PredictionType.Epsilon
                ? GetPredictedSample(modelOutput, sample, sigma)
                : sample.SubtractTensors(modelOutput.MultipleTensorByFloat(sigmaHat));


            // 2. Convert to an ODE derivative
            var derivative = sample
                .SubtractTensors(predOriginalSample)
                .DivideTensorByFloat(sigmaHat);

            var delta = _sigmas[stepIndex + 1] - sigmaHat;
            return new SchedulerStepResult(sample.AddTensors(derivative.MultipleTensorByFloat(delta)));
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
                .Select(x => _sigmas[x])
                .Max();

            return noise
                .MultipleTensorByFloat(sigma)
                .AddTensors(originalSamples);
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected override void Dispose(bool disposing)
        {
            _sigmas = null;
            base.Dispose(disposing);
        }
    }
}

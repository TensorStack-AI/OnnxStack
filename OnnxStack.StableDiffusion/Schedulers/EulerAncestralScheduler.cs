using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers
{
    public sealed class EulerAncestralScheduler : SchedulerBase
    {
        private float[] _sigmas;

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
            var betas = Enumerable.Empty<float>();
            if (!Options.TrainedBetas.IsNullOrEmpty())
            {
                betas = Options.TrainedBetas;
            }
            else if (Options.BetaSchedule == BetaScheduleType.Linear)
            {
                var steps = Options.TrainTimesteps - 1;
                var delta = Options.BetaStart + (Options.BetaEnd - Options.BetaStart);
                betas = Enumerable.Range(0, Options.TrainTimesteps)
                    .Select(i => delta * i / steps);
            }
            else if (Options.BetaSchedule == BetaScheduleType.ScaledLinear)
            {
                var start = (float)Math.Sqrt(Options.BetaStart);
                var end = (float)Math.Sqrt(Options.BetaEnd);
                betas = np.linspace(start, end, Options.TrainTimesteps)
                    .ToArray<float>()
                    .Select(x => x * x);
            }
            else if (Options.BetaSchedule == BetaScheduleType.SquaredCosCapV2)
            {
                betas = GetBetasForAlphaBar();
            }


            var alphas = betas.Select(beta => 1 - beta);
            var cumulativeProduct = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b));

            // Create _sigmas as a list and reverse it
            _sigmas = cumulativeProduct
                .Select(alpha_prod => (float)Math.Sqrt((1 - alpha_prod) / alpha_prod))
                .ToArray();

            // standard deviation of the initial noise distrubution
            var maxSigma = _sigmas.Max();
            var initNoiseSigma = Options.TimestepSpacing == TimestepSpacingType.Linspace || Options.TimestepSpacing == TimestepSpacingType.Trailing
                ? maxSigma
                : (float)Math.Sqrt(maxSigma * maxSigma + 1);
            SetInitNoiseSigma(initNoiseSigma);
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            float[] timesteps = null;
            if (Options.TimestepSpacing == TimestepSpacingType.Linspace)
            {
                float start = 0;
                float stop = Options.TrainTimesteps - 1;
                timesteps = np.around(np.linspace(start, stop, Options.InferenceSteps))
                    .ToArray<float>();
            }
            else if (Options.TimestepSpacing == TimestepSpacingType.Leading)
            {
                int stepRatio = Options.TrainTimesteps / Options.InferenceSteps;
                timesteps = np.around(np.arange(0, Options.InferenceSteps) * stepRatio)
                        // ["::-1"] // Reverse
                        .copy()
                        .astype(NPTypeCode.Single)
                        .ToArray<float>()
                        .Select(x => x + Options.StepsOffset)
                        .ToArray();
            }
            else if (Options.TimestepSpacing == TimestepSpacingType.Trailing)
            {
                int stepRatio = Options.TrainTimesteps / Options.InferenceSteps;
                timesteps = np.around(np.arange(Options.TrainTimesteps, 0, -stepRatio))
                     ["::-1"] // Reverse
                     [":-1"]  // Skip last
                     .copy()
                     .astype(NPTypeCode.Single)
                     .ToArray<float>()
                     .Select(x => x - 1f)
                     .ToArray();
            }


            var sigmas = np.array(_sigmas);
            var log_sigmas = np.log(sigmas);
            var range = np.arange(0, (float)_sigmas.Length).ToArray<float>();
            sigmas = Interpolate(timesteps, range, _sigmas);

            if (Options.UseKarrasSigmas)
            {
                sigmas = ConvertToKarras(sigmas);
                timesteps = SigmaToTimestep(sigmas, log_sigmas).ToArray();
            }

            //  add 0.000 to the end of the result
            sigmas = np.add(sigmas, 0.000f);


            _sigmas = sigmas.ToArray<float>();

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
            sample = sample.DivideTensorByFloat(sigma, sample.Dimensions);

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
        public override DenseTensor<float> Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            var stepIndex = Timesteps.IndexOf(timestep);
            var sigma = _sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = sample.SubtractTensors(modelOutput.MultipleTensorByFloat(sigma));

            var sigmaFrom = _sigmas[stepIndex];
            var sigmaTo = _sigmas[stepIndex + 1];

            var sigmaFromLessSigmaTo = MathF.Pow(sigmaFrom, 2) - MathF.Pow(sigmaTo, 2);
            var sigmaUpResult = MathF.Pow(sigmaTo, 2) * sigmaFromLessSigmaTo / MathF.Pow(sigmaFrom, 2);
            var sigmaUp = sigmaUpResult < 0 ? -MathF.Pow(MathF.Abs(sigmaUpResult), 0.5f) : MathF.Pow(sigmaUpResult, 0.5f);

            var sigmaDownResult = MathF.Pow(sigmaTo, 2) - MathF.Pow(sigmaUp, 2);
            var sigmaDown = sigmaDownResult < 0 ? -MathF.Pow(MathF.Abs(sigmaDownResult), 0.5f) : MathF.Pow(sigmaDownResult, 0.5f);

            // 2. Convert to an ODE derivative
            var derivative = sample
                .SubtractTensors(predOriginalSample)
                .DivideTensorByFloat(sigma, predOriginalSample.Dimensions);

            var delta = sigmaDown - sigma;
            var prevSample = sample.AddTensors(derivative.MultipleTensorByFloat(delta));
            var noise = CreateRandomSample(prevSample.Dimensions);
            return prevSample.AddTensors(noise.MultipleTensorByFloat(sigmaUp));
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
            // TODO: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py#L389
            var sigma = timesteps
                .Select(x => Timesteps.IndexOf(x))
                .Select(x => _sigmas[x])
                .Max();

            return noise
                .MultipleTensorByFloat(sigma)
                .AddTensors(originalSamples);
        }

        protected override void Dispose(bool disposing)
        {
            _sigmas = null;
            base.Dispose(disposing);
        }
    }
}

using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers
{
    public abstract class SchedulerBase : IScheduler
    {
        private readonly Random _random;
        private readonly List<int> _timesteps;
        private readonly SchedulerOptions _options;
        private float _initNoiseSigma = 1f;

        /// <summary>
        /// Initializes a new instance of the <see cref="SchedulerBase"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public SchedulerBase(SchedulerOptions schedulerOptions)
        {
            _options = schedulerOptions;
            _random = new Random(_options.Seed);
            Initialize();
            _timesteps = [.. SetTimesteps()];
        }

        /// <summary>
        /// Gets or sets the sigmas.
        /// </summary>
        protected float[] Sigmas { get; set; }

        /// <summary>
        /// Gets or sets the alphas.
        /// </summary>
        protected float[] Alphas { get; set; }

        /// <summary>
        /// Gets or sets the betas.
        /// </summary>
        protected float[] Betas { get; set; }

        /// <summary>
        /// Gets or sets the alphas cum product.
        /// </summary>
        protected float[] AlphasCumProd { get; set; }

        /// <summary>
        /// Gets the scheduler options.
        /// </summary>
        public SchedulerOptions Options => _options;

        /// <summary>
        /// Gets the random initiated with the seed.
        /// </summary>
        public Random Random => _random;

        /// <summary>
        /// Gets the initial noise sigma.
        /// </summary>
        public float InitNoiseSigma => _initNoiseSigma;

        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        public IReadOnlyList<int> Timesteps => _timesteps;

        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="sample">The sample.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public abstract DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep);

        /// <summary>
        /// Processes a inference step for the specified model output.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="order">The order.</param>
        /// <returns></returns>
        public abstract SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int contextSize = 16);

        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="originalSamples">The original samples.</param>
        /// <param name="noise">The noise.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        public abstract DenseTensor<float> AddNoise(DenseTensor<float> originalSamples, DenseTensor<float> noise, IReadOnlyList<int> timesteps);

        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected abstract int[] SetTimesteps();


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected virtual void Initialize()
        {
            Betas = GetBetaSchedule();
            Alphas = Betas
                .Select(beta => 1.0f - beta)
                .ToArray();
            AlphasCumProd = Alphas
                .Select((alpha, i) => Alphas.Take(i + 1).Aggregate((a, b) => a * b))
                .ToArray();
            Sigmas = AlphasCumProd
                 .Select(alpha_prod => MathF.Sqrt((1f - alpha_prod) / alpha_prod))
                 .OrderByDescending(x => x)
                 .ToArray();
        }


        /// <summary>
        /// Creates a random sample with the specified dimesions.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public virtual DenseTensor<float> CreateRandomSample(ReadOnlySpan<int> dimensions, float initialNoiseSigma = 1f)
        {
            return Random.NextTensor(dimensions, initialNoiseSigma);
        }


        /// <summary>
        /// Gets the beta schedule.
        /// </summary>
        /// <returns></returns>
        protected virtual float[] GetBetaSchedule()
        {
            var betas = Enumerable.Empty<float>();
            if (Options.TrainedBetas != null)
            {
                betas = Options.TrainedBetas;
            }
            else if (Options.BetaSchedule == BetaScheduleType.Linear)
            {
                betas = ArrayHelpers.Linspace(Options.BetaStart, Options.BetaEnd, Options.TrainTimesteps);
            }
            else if (Options.BetaSchedule == BetaScheduleType.ScaledLinear)
            {
                var start = MathF.Sqrt(Options.BetaStart);
                var end = MathF.Sqrt(Options.BetaEnd);
                betas = ArrayHelpers.Linspace(start, end, Options.TrainTimesteps).Select(x => x * x);
            }
            else if (Options.BetaSchedule == BetaScheduleType.SquaredCosCapV2)
            {
                betas = GetBetasForAlphaBar();
            }
            else if (Options.BetaSchedule == BetaScheduleType.Sigmoid)
            {
                var mul = Options.BetaEnd - Options.BetaStart;
                var betaSig = ArrayHelpers.Linspace(-6f, 6f, Options.TrainTimesteps);
                var sigmoidBetas = betaSig
                    .Select(beta => 1.0f / (1.0f + MathF.Exp(-beta)))
                    .ToArray();
                betas = sigmoidBetas
                    .Select(x => (x * mul) + Options.BetaStart)
                    .ToArray();
            }
            return betas.ToArray();
        }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected virtual float[] GetTimesteps()
        {
            if (Options.TimestepSpacing == TimestepSpacingType.Linspace)
            {
                if (Options.InferenceSteps <= 1)
                    return [Options.TrainTimesteps - 1];

                return ArrayHelpers.Linspace(0, Options.TrainTimesteps - 1, Options.InferenceSteps)
                    .OrderByDescending(x => x)
                    .ToArray();
            }
            else if (Options.TimestepSpacing == TimestepSpacingType.Leading)
            {
                var stepRatio = Options.TrainTimesteps / Options.InferenceSteps;
                return Enumerable.Range(0, Options.InferenceSteps)
                    .Select(x => ((float)x * stepRatio) + Options.StepsOffset)
                    .OrderByDescending(x => x)
                    .ToArray();
            }
            else if (Options.TimestepSpacing == TimestepSpacingType.Trailing)
            {
                var stepRatio = Options.TrainTimesteps / Math.Max(1, Options.InferenceSteps);
                var result = Enumerable.Range(0, Options.TrainTimesteps + 1)
                    .Where((number, index) => index % stepRatio == 0 && number > 0)
                    .Select(x => MathF.Max(0, x - 1f))
                    .OrderByDescending(x => x)
                    .ToArray();
                return result;
            }

            throw new NotImplementedException();
        }


        /// <summary>
        /// Gets the predicted sample.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="sigma">The sigma.</param>
        /// <returns></returns>
        protected virtual DenseTensor<float> GetPredictedSample(DenseTensor<float> modelOutput, DenseTensor<float> sample, float sigma)
        {
            DenseTensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                predOriginalSample = sample.SubtractTensors(modelOutput.MultiplyTensorByFloat(sigma));
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                var sigmaSqrt = MathF.Sqrt(sigma * sigma + 1);
                predOriginalSample = sample.DivideTensorByFloat(sigmaSqrt)
                    .AddTensors(modelOutput.MultiplyTensorByFloat(-sigma / sigmaSqrt));
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                //prediction_type not implemented yet: sample
                predOriginalSample = sample.CloneTensor();
            }
            return predOriginalSample;
        }


        /// <summary>
        /// Sets the initial noise sigma.
        /// </summary>
        /// <param name="initNoiseSigma">The initial noise sigma.</param>
        protected void SetInitNoiseSigma()
        {
            var maxSigma = Sigmas.Max();
            var initNoiseSigma = Options.TimestepSpacing == TimestepSpacingType.Linspace || Options.TimestepSpacing == TimestepSpacingType.Trailing
                ? maxSigma 
                : MathF.Sqrt(maxSigma * maxSigma + 1f);
            _initNoiseSigma = initNoiseSigma;
        }


        /// <summary>
        /// Gets the previous timestep.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected int GetPreviousTimestep(int timestep)
        {
            var index = Timesteps.IndexOf(timestep) + 1;
            if (index > Timesteps.Count - 1)
                return 0;

            return Timesteps[index];
        }


        /// <summary>
        /// Gets the betas for alpha bar.
        /// </summary>
        /// <param name="maxBeta">The maximum beta.</param>
        /// <param name="alphaTransformType">Type of the alpha transform.</param>
        /// <returns></returns>
        protected float[] GetBetasForAlphaBar()
        {
            Func<float, float> alphaBarFn = null;
            if (_options.AlphaTransformType == AlphaTransformType.Cosine)
            {
                alphaBarFn = t => MathF.Pow(MathF.Cos((t + 0.008f) / 1.008f * MathF.PI / 2.0f), 2.0f);
            }
            else if (_options.AlphaTransformType == AlphaTransformType.Exponential)
            {
                alphaBarFn = t => MathF.Exp(t * -12.0f);
            }

            return Enumerable
                .Range(0, _options.TrainTimesteps)
                .Select(i =>
                {
                    var t1 = (float)i / _options.TrainTimesteps;
                    var t2 = (float)(i + 1) / _options.TrainTimesteps;
                    return MathF.Min(1f - alphaBarFn(t2) / alphaBarFn(t1), _options.MaximumBeta);
                }).ToArray();
        }


        /// <summary>
        /// Interpolates the specified timesteps.
        /// </summary>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="range">The range.</param>
        /// <param name="sigmas">The sigmas.</param>
        /// <returns></returns>
        protected virtual float[] Interpolate(float[] timesteps, float[] range, float[] sigmas)
        {
            var result = new float[timesteps.Length];
            for (int i = 0; i < timesteps.Length; i++)
            {
                float t = timesteps[i];
                int index = ArrayHelpers.BinarySearchDescending(range, t);
                if (index >= 0)
                {
                    // Exact match
                    result[i] = sigmas[index];
                }
                else
                {
                    index = ~index;
                    if (index == 0)
                    {
                        // t < range[0], clamp to first
                        result[i] = sigmas[0];
                    }
                    else if (index >= range.Length)
                    {
                        // t > range[^1], clamp to last
                        result[i] = sigmas[^1];
                    }
                    else
                    {
                        // Interpolate between index - 1 and index
                        float t0 = range[index - 1];
                        float t1 = range[index];
                        float s0 = sigmas[index - 1];
                        float s1 = sigmas[index];
                        float factor = (t - t0) / (t1 - t0);
                        result[i] = s0 + factor * (s1 - s0);
                    }
                }
            }

            return result;
        }


        /// <summary>
        /// Converts sigmas to karras.
        /// </summary>
        /// <param name="inSigmas">The in sigmas.</param>
        /// <returns></returns>
        protected float[] ConvertToKarras(float[] inSigmas)
        {
            // Get the minimum and maximum values from the input sigmas
            float sigmaMin = inSigmas[^1];
            float sigmaMax = inSigmas[0];

            // Set the value of rho, which is used in the calculation
            float rho = 7.0f; // 7.0 is the value used in the paper

            // Create a linear ramp from 0 to 1
            float[] ramp = Enumerable.Range(0, _options.InferenceSteps)
                .Select(i => (float)i / (_options.InferenceSteps - 1))
                .ToArray();

            // Calculate the inverse of sigmaMin and sigmaMax raised to the power of 1/rho
            float minInvRho = MathF.Pow(sigmaMin, 1.0f / rho);
            float maxInvRho = MathF.Pow(sigmaMax, 1.0f / rho);

            // Calculate the Karras noise schedule using the formula from the paper
            float[] sigmas = new float[_options.InferenceSteps];
            for (int i = 0; i < _options.InferenceSteps; i++)
            {
                sigmas[i] = MathF.Pow(maxInvRho + ramp[i] * (minInvRho - maxInvRho), rho);
            }
            return sigmas;
        }


        /// <summary>
        /// Create timesteps form sigmas
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <param name="logSigmas">The log sigmas.</param>
        /// <returns></returns>
        protected float[] SigmaToTimestep(float[] sigmas2, float[] logSigmas2)
        {
            var sigmas = sigmas2.Reverse().ToArray();
            var logSigmas = logSigmas2.Reverse().ToArray();

            var timesteps = new float[sigmas.Length];
            for (int i = 0; i < sigmas.Length; i++)
            {
                float logSigma = MathF.Log(sigmas[i].ZeroIfNan());
                float[] dists = new float[logSigmas.Length];

                for (int j = 0; j < logSigmas.Length; j++)
                {
                    dists[j] = logSigma - logSigmas[j];
                }

                int lowIdx = 0;
                int highIdx = 1;

                for (int j = 0; j < logSigmas.Length - 1; j++)
                {
                    if (dists[j] >= 0)
                    {
                        lowIdx = j;
                        highIdx = j + 1;
                    }
                }

                float low = logSigmas[lowIdx];
                float high = logSigmas[highIdx];

                float w = (low - logSigma) / (low - high);
                w = Math.Clamp(w, 0, 1);

                float ti = (1 - w) * lowIdx + w * highIdx;
                timesteps[i] = ti;
            }

            return timesteps;
        }

        #region IDisposable

        private bool disposed = false;


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                // Dispose managed resources here.
                _timesteps?.Clear();
                Sigmas = null;
                Alphas = null;
                Betas = null;
                AlphasCumProd = null;
            }

            // Dispose unmanaged resources here (if any).
            disposed = true;
        }


        /// <summary>
        /// Finalizes an instance of the <see cref="SchedulerBase"/> class.
        /// </summary>
        ~SchedulerBase()
        {
            Dispose(false);
        }

        #endregion
    }
}
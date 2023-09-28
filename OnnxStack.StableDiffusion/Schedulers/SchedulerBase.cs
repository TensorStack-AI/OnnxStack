using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using NumSharp.Generic;
using OnnxStack.StableDiffusion.Config;
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
        private float _initNoiseSigma;

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
            _timesteps = new List<int>(SetTimesteps());
        }

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
        public abstract DenseTensor<float> Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4);


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="originalSamples">The original samples.</param>
        /// <param name="noise">The noise.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        public abstract DenseTensor<float> AddNoise(DenseTensor<float> originalSamples, DenseTensor<float> noise, int[] timesteps);

        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected abstract void Initialize();

        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected abstract int[] SetTimesteps();


        /// <summary>
        /// Creates a random sample with the specified dimesions.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns></returns>
        public virtual DenseTensor<float> CreateRandomSample(ReadOnlySpan<int> dimensions, float initialNoiseSigma = 1f)
        {
            var latents = new DenseTensor<float>(dimensions);
            for (int i = 0; i < latents.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = _random.NextDouble(); // Uniform(0,1) random number
                var u2 = _random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number
                latents.SetValue(i, (float)standardNormalRand * initialNoiseSigma);
            }
            return latents;
        }


        /// <summary>
        /// Sets the initial noise sigma.
        /// </summary>
        /// <param name="initNoiseSigma">The initial noise sigma.</param>
        protected void SetInitNoiseSigma(float initNoiseSigma)
        {
            _initNoiseSigma = initNoiseSigma;
        }


        /// <summary>
        /// Gets the previous timestep.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected int GetPreviousTimestep(int timestep)
        {
            return timestep - _options.TrainTimesteps / _options.InferenceSteps;
        }


        /// <summary>
        /// Gets the betas for alpha bar.
        /// </summary>
        /// <param name="maxBeta">The maximum beta.</param>
        /// <param name="alphaTransformType">Type of the alpha transform.</param>
        /// <returns></returns>
        protected float[] GetBetasForAlphaBar()
        {
            var betas = new float[_options.TrainTimesteps];

            Func<float, float> alphaBarFn = null;
            if (_options.AlphaTransformType == AlphaTransformType.Cosine)
            {
                alphaBarFn = t => (float)Math.Pow(Math.Cos((t + 0.008) / 1.008 * Math.PI / 2.0), 2.0);
            }
            else if (_options.AlphaTransformType == AlphaTransformType.Exponential)
            {
                alphaBarFn = t => (float)Math.Exp(t * -12.0);
            }
   
            for (int i = 0; i < _options.TrainTimesteps; i++)
            {
                float t1 = (float)i / _options.TrainTimesteps;
                float t2 = (float)(i + 1) / _options.TrainTimesteps;
                float alphaT1 = alphaBarFn(t1);
                float alphaT2 = alphaBarFn(t2);
                float beta = Math.Min(1 - alphaT2 / alphaT1, _options.MaximumBeta);
                betas[i] = (float)Math.Max(beta, 0.0001);
            }
            return betas;
        }


        /// <summary>
        /// Interpolates the specified timesteps.
        /// </summary>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="range">The range.</param>
        /// <param name="sigmas">The sigmas.</param>
        /// <returns></returns>
        protected NDArray Interpolate(float[] timesteps, float[] range, float[] sigmas)
        {
            // Create an output array with the same shape as timesteps
            var result = np.zeros(Shape.Vector(timesteps.Length + 1), NPTypeCode.Single);

            // Loop over each element of timesteps
            for (int i = 0; i < timesteps.Length; i++)
            {
                // Find the index of the first element in range that is greater than or equal to timesteps[i]
                int index = Array.BinarySearch(range, timesteps[i]);

                // If timesteps[i] is exactly equal to an element in range, use the corresponding value in sigma
                if (index >= 0)
                {
                    result[i] = sigmas[(sigmas.Length - 1) - index];
                }

                // If timesteps[i] is less than the first element in range, use the first value in sigmas
                else if (index == -1)
                {
                    result[i] = sigmas[sigmas.Length - 1];
                }

                // If timesteps[i] is greater than the last element in range, use the last value in sigmas
                else if (index == -range.Length - 1)
                {
                    result[i] = sigmas[0];
                }

                // Otherwise, interpolate linearly between two adjacent values in sigmas
                else
                {
                    index = ~index; // bitwise complement of j gives the insertion point of x[i]
                    var startIndex = (sigmas.Length - 1) - index;
                    float t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]); // fractional distance between two points
                    result[i] = sigmas[startIndex - 1] + t * (sigmas[startIndex] - sigmas[startIndex - 1]); // linear interpolation formula
                }
            }
            return result;
        }


        /// <summary>
        /// Converts sigmas to karras.
        /// </summary>
        /// <param name="inSigmas">The in sigmas.</param>
        /// <returns></returns>
        protected NDArray ConvertToKarras(NDArray inSigmas)
        {
            // Get the minimum and maximum values from the input sigmas
            float sigmaMin = inSigmas[inSigmas.size - 1];
            float sigmaMax = inSigmas[0];

            // Set the value of rho, which is used in the calculation
            float rho = 7.0f; // 7.0 is the value used in the paper

            // Create a linear ramp from 0 to 1
            float[] ramp = Enumerable.Range(0, _options.InferenceSteps)
                .Select(i => (float)i / (_options.InferenceSteps - 1))
                .ToArray();

            // Calculate the inverse of sigmaMin and sigmaMax raised to the power of 1/rho
            float minInvRho = (float)Math.Pow(sigmaMin, 1.0 / rho);
            float maxInvRho = (float)Math.Pow(sigmaMax, 1.0 / rho);

            // Calculate the Karras noise schedule using the formula from the paper
            float[] sigmas = new float[_options.InferenceSteps];
            for (int i = 0; i < _options.InferenceSteps; i++)
            {
                sigmas[i] = (float)Math.Pow(maxInvRho + ramp[i] * (minInvRho - maxInvRho), rho);
            }

            // Return the resulting noise schedule as a Vector<float>
            return sigmas;
        }


        /// <summary>
        /// Create timesteps form sigmas
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <param name="logSigmas">The log sigmas.</param>
        /// <returns></returns>
        protected float[] SigmaToTimestep(NDArray sigmas, NDArray logSigmas)
        {
            int numSigmas = sigmas.size;
            int numLogSigmas = logSigmas.size;
            var floatSigmas = sigmas.view<float>();
            var floatLogSigmas = logSigmas.view<float>();

            NDArray<float> t = new NDArray<float>(numSigmas);

            for (int i = 0; i < numSigmas; i++)
            {
                float logSigma = (float)Math.Log(floatSigmas[i]);
                float[] dists = new float[numLogSigmas];

                for (int j = 0; j < numLogSigmas; j++)
                {
                    dists[j] = logSigma - floatLogSigmas[j];
                }

                int lowIdx = 0;
                int highIdx = 1;

                for (int j = 0; j < numLogSigmas - 1; j++)
                {
                    if (dists[j] >= 0)
                    {
                        lowIdx = j;
                        highIdx = j + 1;
                    }
                }

                float low = floatLogSigmas[lowIdx];
                float high = floatLogSigmas[highIdx];

                float w = (low - logSigma) / (low - high);
                w = Math.Clamp(w, 0, 1);

                float ti = (1 - w) * lowIdx + w * highIdx;
                t[i] = ti;
            }

            return t.ToArray<float>();
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
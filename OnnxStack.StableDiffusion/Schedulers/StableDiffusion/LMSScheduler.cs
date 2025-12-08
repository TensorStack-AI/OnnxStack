using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    public class LMSScheduler : SchedulerBase
    {
        private int _order = 4;
        private Queue<DenseTensor<float>> _derivatives;

        /// <summary>
        /// Initializes a new instance of the <see cref="LMSScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public LMSScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMSScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public LMSScheduler(SchedulerOptions schedulerOptions) : base(schedulerOptions) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            base.Initialize();
            _derivatives = new Queue<DenseTensor<float>>();
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
            int stepIndex = Timesteps.IndexOf(timestep);
            var sigma = Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = GetPredictedSample(modelOutput, sample, sigma);

            // 2. Convert to an ODE derivative
            var derivativeSample = sample
                .SubtractTensors(predOriginalSample)
                .DivideTensorByFloat(sigma);

            _derivatives.Enqueue(derivativeSample);
            if (_derivatives.Count > _order)
                _derivatives.Dequeue();

            // 3. compute linear multistep coefficients
            _order = Math.Min(stepIndex + 1, _order);
            var lmsCoeffs = Enumerable.Range(0, _order)
                .Select(currOrder => GetLmsCoefficient(_order, stepIndex, currOrder));

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives
            var revDerivatives = _derivatives.Reverse();

            // Create list of tuples from the lmsCoeffs and reversed derivatives
            var lmsCoeffsAndDerivatives = lmsCoeffs
                .Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative))
                .ToArray();

            // Create tensor for product of lmscoeffs and derivatives
            var lmsDerProduct = new DenseTensor<float>[_derivatives.Count];
            for (int i = 0; i < lmsCoeffsAndDerivatives.Length; i++)
            {
                // Multiply to coeff by each derivatives to create the new tensors
                var (lmsCoeff, derivative) = lmsCoeffsAndDerivatives[i];
                lmsDerProduct[i] = derivative.MultiplyTensorByFloat(lmsCoeff);
            }

            // Add the sumed tensor to the sample
            return new SchedulerStepResult(sample.AddTensors(lmsDerProduct.SumTensors(modelOutput.Dimensions)));
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
            // Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py#L439
            var sigma = timesteps
                .Select(x => Timesteps.IndexOf(x))
                .Select(x => Sigmas[x])
                .Max();

            return noise
                .MultiplyTensorByFloat(sigma)
                .AddTensors(originalSamples);
        }


        /// <summary>
        /// Gets the LMS coefficient.
        /// </summary>
        /// <param name="order">The order.</param>
        /// <param name="t">The t.</param>
        /// <param name="currentOrder">The current order.</param>
        /// <returns></returns>
        private float GetLmsCoefficient(int order, int t, int currentOrder)
        {
            return MathHelpers.IntegrateOnClosedInterval(tau => GetLmsDerivative(tau, order, t, currentOrder), Sigmas[t], Sigmas[t + 1], 1e-4);
        }


        /// <summary>
        /// LMSs the derivative.
        /// </summary>
        /// <param name="tau">The tau.</param>
        /// <param name="order">The order.</param>
        /// <param name="t">The t.</param>
        /// <param name="currentOrder">The current order.</param>
        /// <returns>System.Double.</returns>
        private double GetLmsDerivative(double tau, int order, int t, int currentOrder)
        {
            double prod = 1.0;
            for (int k = 0; k < order; k++)
            {
                if (currentOrder == k)
                    continue;

                prod *= (tau - Sigmas[t - k]) / (Sigmas[t - currentOrder] - Sigmas[t - k]);
            }
            return prod;
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected override void Dispose(bool disposing)
        {
            _derivatives?.Clear();
            base.Dispose(disposing);
        }
    }
}

using MathNet.Numerics;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers
{
    public sealed class LMSScheduler : SchedulerBase
    {
        private float[] _sigmas;
        private readonly List<DenseTensor<float>> _derivatives;

        /// <summary>
        /// Initializes a new instance of the <see cref="LMSScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public LMSScheduler(StableDiffusionOptions stableDiffusionOptions) 
            : this(stableDiffusionOptions, new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMSScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public LMSScheduler(StableDiffusionOptions stableDiffusionOptions, SchedulerOptions schedulerOptions) 
            : base(stableDiffusionOptions, schedulerOptions)
        {
            _derivatives = new List<DenseTensor<float>>();
        }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            var betas = Enumerable.Empty<float>();
            if (!SchedulerOptions.TrainedBetas.IsNullOrEmpty())
            {
                betas = SchedulerOptions.TrainedBetas;
            }
            else if (SchedulerOptions.BetaSchedule == BetaSchedule.Linear)
            {
                var steps = SchedulerOptions.TrainTimesteps - 1;
                var delta = SchedulerOptions.BetaStart + (SchedulerOptions.BetaEnd - SchedulerOptions.BetaStart);
                betas = Enumerable.Range(0, SchedulerOptions.TrainTimesteps)
                    .Select(i => delta * i / steps);
            }
            else if (SchedulerOptions.BetaSchedule == BetaSchedule.ScaledLinear)
            {
                var start = (float)Math.Sqrt(SchedulerOptions.BetaStart);
                var end = (float)Math.Sqrt(SchedulerOptions.BetaEnd);
                betas = np.linspace(start, end, SchedulerOptions.TrainTimesteps)
                    .ToArray<float>()
                    .Select(x => x * x);
            }
            else if (SchedulerOptions.BetaSchedule == BetaSchedule.SquaredCosCapV2)
            {
                betas = GetBetasForAlphaBar();
            }


            var alphas = betas.Select(beta => 1 - beta);
            var cumulativeProduct = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b));

            // Create sigmas as a list and reverse it
            _sigmas = cumulativeProduct
                .Select(alpha_prod => (float)Math.Sqrt((1 - alpha_prod) / alpha_prod))
                .ToArray();

            // standard deviation of the initial noise distrubution
            var maxSigma = _sigmas.Max();
            var initNoiseSigma = SchedulerOptions.TimestepSpacing == TimestepSpacing.Linspace || SchedulerOptions.TimestepSpacing == TimestepSpacing.Trailing
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
            if (SchedulerOptions.TimestepSpacing == TimestepSpacing.Linspace)
            {
                float start = 0;
                float stop = SchedulerOptions.TrainTimesteps - 1;
                timesteps = np.around(np.linspace(start, stop, StableDiffusionOptions.NumInferenceSteps))
                    .ToArray<float>();
            }
            else if (SchedulerOptions.TimestepSpacing == TimestepSpacing.Leading)
            {
                int stepRatio = SchedulerOptions.TrainTimesteps / StableDiffusionOptions.NumInferenceSteps;
                timesteps = np.around(np.arange(0, StableDiffusionOptions.NumInferenceSteps) * stepRatio)
                        // ["::-1"] // Reverse
                        .copy()
                        .astype(NPTypeCode.Single)
                        .ToArray<float>()
                        .Select(x => x + SchedulerOptions.StepsOffset)
                        .ToArray();
            }
            else if (SchedulerOptions.TimestepSpacing == TimestepSpacing.Trailing)
            {
                int stepRatio = SchedulerOptions.TrainTimesteps / StableDiffusionOptions.NumInferenceSteps;
                timesteps = np.around(np.arange(SchedulerOptions.TrainTimesteps, 0, -stepRatio))
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

            if (SchedulerOptions.UseKarrasSigmas)
            {
                sigmas = ConvertToKarras(sigmas);
                timesteps = SigmaToTimestep(sigmas, log_sigmas);
            }

            //  add 0.000 to the end of the result
            sigmas = np.add(sigmas, 0.000f);

            _sigmas = sigmas.ToArray<float>();
            return timesteps.Select(x => (int)x)
                 .OrderByDescending(x => x)
                 .ToArray();
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
            int stepIndex = Timesteps.IndexOf(timestep);
            var sigma = _sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            var predOriginalSample = TensorHelper.SubtractTensors(sample, TensorHelper.MultipleTensorByFloat(modelOutput, sigma));

            // 2. Convert to an ODE derivative
            var derivativeSample = TensorHelper.DivideTensorByFloat(TensorHelper.SubtractTensors(sample, predOriginalSample), sigma, sample.Dimensions);

            _derivatives.Add(derivativeSample);
            if (_derivatives.Count > order)
            {
                // remove first element
                _derivatives.RemoveAt(0);
            }

            // 3. compute linear multistep coefficients
            order = Math.Min(stepIndex + 1, order);
            var lmsCoeffs = Enumerable.Range(0, order).Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder));

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives
            var revDerivatives = Enumerable.Reverse(_derivatives);

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
                lmsDerProduct[i] = TensorHelper.MultipleTensorByFloat(derivative, (float)lmsCoeff);
            }

            // Sum the tensors
            var sumTensor = TensorHelper.SumTensors(lmsDerProduct, StableDiffusionOptions.GetScaledDimension());

            // Add the sumed tensor to the sample
            return TensorHelper.AddTensors(sample, sumTensor);
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
            sample = TensorHelper.DivideTensorByFloat(sample, sigma, sample.Dimensions);

            return sample;
        }


        /// <summary>
        /// Gets the LMS coefficient.
        /// </summary>
        /// <param name="order">The order.</param>
        /// <param name="t">The t.</param>
        /// <param name="currentOrder">The current order.</param>
        /// <returns></returns>
        private double GetLmsCoefficient(int order, int t, int currentOrder)
        {  //python line 135 of scheduling_lms_discrete.py
            // Compute a linear multistep coefficient.
            double LmsDerivative(double tau)
            {
                double prod = 1.0;
                for (int k = 0; k < order; k++)
                {
                    if (currentOrder == k)
                    {
                        continue;
                    }
                    prod *= (tau - _sigmas[t - k]) / (_sigmas[t - currentOrder] - _sigmas[t - k]);
                }
                return prod;
            }
            return Integrate.OnClosedInterval(LmsDerivative, _sigmas[t], _sigmas[t + 1], 1e-4);
        }
    }
}

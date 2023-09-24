using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers
{
    public abstract class SchedulerBase
    {
        protected readonly Random _random;
        protected readonly SchedulerOptions _schedulerOptions;
        protected readonly StableDiffusionOptions _stableDiffusionOptions;

        protected List<int> _timesteps;
        protected float _initNoiseSigma;
        protected bool _isScaleInputCalled;
        protected DenseTensor<float> _sigmasTensor;
        protected List<float> _alphasCumulativeProducts;
        protected List<double> _computedSigmas;

        public SchedulerBase(StableDiffusionOptions stableDiffusionOptions, SchedulerOptions schedulerOptions)
        {
            _schedulerOptions = schedulerOptions;
            _stableDiffusionOptions = stableDiffusionOptions;
            _random = new Random(_stableDiffusionOptions.Seed);
            Initialize();
        }

        public float GetInitNoiseSigma() => _initNoiseSigma;

        public abstract DenseTensor<float> Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4);

        public virtual int[] SetTimesteps(int inferenceSteps)
        {
            double start = 0;
            double stop = _schedulerOptions.TrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, inferenceSteps).ToArray<double>();

            _timesteps = timesteps.Select(x => (int)x)
                .Reverse()
                .ToList();

            var range = np.arange(0, (double)_computedSigmas.Count).ToArray<double>();
            var sigmas = Interpolate(timesteps, range, _computedSigmas);
            _sigmasTensor = new DenseTensor<float>(sigmas.Length);
            for (int i = 0; i < sigmas.Length; i++)
            {
                _sigmasTensor.SetValue(i, (float)sigmas[i]);
            }
            return _timesteps.ToArray();
        }


        public DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            // Get step index of timestep from TimeSteps
            int stepIndex = _timesteps.IndexOf(timestep);

            // Get sigma at stepIndex
            var sigma = _sigmasTensor[stepIndex];
            sigma = (float)Math.Sqrt(Math.Pow(sigma, 2) + 1);

            // Divide sample tensor shape {2,4,(H/8),(W/8)} by sigma
            sample = TensorHelper.DivideTensorByFloat(sample, sigma, sample.Dimensions);
            _isScaleInputCalled = true;
            return sample;
        }

        protected virtual void Initialize()
        {
            var alphas = new List<float>();
            var betas = new List<float>();

            if (_schedulerOptions.TrainedBetas != null)
            {
                betas = _schedulerOptions.TrainedBetas.ToList();
            }
            else if (_schedulerOptions.BetaSchedule == SchedulerBetaSchedule.Linear)
            {
                betas = Enumerable.Range(0, _schedulerOptions.TrainTimesteps)
                    .Select(i => _schedulerOptions.BetaStart + (_schedulerOptions.BetaEnd - _schedulerOptions.BetaStart) * i / (_schedulerOptions.TrainTimesteps - 1))
                    .ToList();
            }
            else if (_schedulerOptions.BetaSchedule == SchedulerBetaSchedule.ScaledLinear)
            {
                var start = (float)Math.Sqrt(_schedulerOptions.BetaStart);
                var end = (float)Math.Sqrt(_schedulerOptions.BetaEnd);
                betas = np.linspace(start, end, _schedulerOptions.TrainTimesteps)
                    .ToArray<float>()
                    .Select(x => x * x)
                    .ToList();
            }


            alphas = betas.Select(beta => 1 - beta).ToList();

            _alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();

            // Create sigmas as a list and reverse it
            _computedSigmas = _alphasCumulativeProducts
                .Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod))
                .Reverse()
                .ToList();

            // standard deviation of the initial noise distrubution
            _initNoiseSigma = (float)_computedSigmas.Max();
        }

        protected double[] Interpolate(double[] timesteps, double[] range, List<double> sigmas)
        {
            // Create an output array with the same shape as timesteps
            var result = np.zeros(timesteps.Length + 1);

            // Loop over each element of timesteps
            for (int i = 0; i < timesteps.Length; i++)
            {
                // Find the index of the first element in range that is greater than or equal to timesteps[i]
                int index = Array.BinarySearch(range, timesteps[i]);

                // If timesteps[i] is exactly equal to an element in range, use the corresponding value in sigma
                if (index >= 0)
                {
                    result[i] = sigmas[index];
                }

                // If timesteps[i] is less than the first element in range, use the first value in sigmas
                else if (index == -1)
                {
                    result[i] = sigmas[0];
                }

                // If timesteps[i] is greater than the last element in range, use the last value in sigmas
                else if (index == -range.Length - 1)
                {
                    result[i] = sigmas[sigmas.Count - 1];
                }

                // Otherwise, interpolate linearly between two adjacent values in sigmas
                else
                {
                    index = ~index; // bitwise complement of j gives the insertion point of x[i]
                    double t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]); // fractional distance between two points
                    result[i] = sigmas[index - 1] + t * (sigmas[index] - sigmas[index - 1]); // linear interpolation formula
                }
            }

            //  add 0.000 to the end of the result
            result = np.add(result, 0.000f);

            return result.ToArray<double>();
        }
    }
}
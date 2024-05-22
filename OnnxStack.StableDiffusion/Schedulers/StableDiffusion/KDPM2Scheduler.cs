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
    internal class KDPM2Scheduler : SchedulerBase
    {
        private int _stepIndex;
        private float[] _sigmas;
        private float[] _sigmasInterpol;
        private float[] _alphasCumProd;
        private DenseTensor<float> _sample;

        /// <summary>
        /// Initializes a new instance of the <see cref="KDPM2Scheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public KDPM2Scheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="KDPM2Scheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public KDPM2Scheduler(SchedulerOptions options) : base(options) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            _stepIndex = 0;
            _sample = null;
            _alphasCumProd = null;

            var betas = GetBetaSchedule();
            var alphas = betas.Select(beta => 1.0f - beta);
            _alphasCumProd = alphas
                .Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b))
                .ToArray();
            _sigmas = _alphasCumProd
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
            // Create timesteps based on the specified strategy
            var sigmas = _sigmas.ToArray();
            var timesteps = GetTimesteps();
            var logSigmas = ArrayHelpers.Log(sigmas);
            var range = ArrayHelpers.Range(0, _sigmas.Length);
            sigmas = Interpolate(timesteps, range, _sigmas);

            if (Options.UseKarrasSigmas)
            {
                sigmas = ConvertToKarras(sigmas);
                timesteps = SigmaToTimestep(sigmas, logSigmas);
            }

            //# interpolate sigmas
            var sigmasInterpol = InterpolateSigmas(sigmas);

            _sigmas = Interleave(sigmas);
            _sigmasInterpol = Interleave(sigmasInterpol);

            var timestepsInterpol = SigmaToTimestep(sigmasInterpol, logSigmas);
            var interleavedTimesteps = timestepsInterpol
                .Concat(timesteps)
                .Select(x => (int)x)
                .OrderByDescending(x => x)
                .ToArray();
            return interleavedTimesteps;
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="sample">The sample.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public override DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            var sigma = _sample is null
                ? _sigmas[_stepIndex]
                : _sigmasInterpol[_stepIndex];

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
        /// <exception cref="ArgumentException">Invalid prediction_type: {SchedulerOptions.PredictionType}</exception>
        /// <exception cref="NotImplementedException">KDPM2Scheduler Thresholding currently not implemented</exception>
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            float sigma;
            float sigmaInterpol;
            float sigmaNext;
            bool isFirstPass = _sample is null;
            if (isFirstPass)
            {
                sigma = _sigmas[_stepIndex];
                sigmaInterpol = _sigmasInterpol[_stepIndex + 1];
                sigmaNext = _sigmas[_stepIndex + 1];
            }
            else
            {
                sigma = _sigmas[_stepIndex - 1];
                sigmaInterpol = _sigmasInterpol[_stepIndex];
                sigmaNext = _sigmas[_stepIndex];
            }

            //# currently only gamma=0 is supported. This usually works best anyways.
            float gamma = 0f;
            float sigmaHat = sigma * (gamma + 1f);
            var sigmaInput = isFirstPass ? sigmaHat : sigmaInterpol;
            DenseTensor<float> predOriginalSample;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                predOriginalSample = sample.SubtractTensors(modelOutput.MultiplyTensorByFloat(sigmaInput));
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                var sigmaSqrt = (float)Math.Sqrt(sigmaInput * sigmaInput + 1f);
                predOriginalSample = sample.DivideTensorByFloat(sigmaSqrt)
                    .AddTensors(modelOutput.MultiplyTensorByFloat(-sigmaInput / sigmaSqrt));
            }
            else
            {
                predOriginalSample = modelOutput.ToDenseTensor();
            }


            float dt;
            DenseTensor<float> derivative;
            if (isFirstPass)
            {
                dt = sigmaInterpol - sigmaHat;
                derivative = sample
                    .SubtractTensors(predOriginalSample)
                    .DivideTensorByFloat(sigmaHat);
                _sample = sample.ToDenseTensor();
            }
            else
            {
                dt = sigmaNext - sigmaHat;
                derivative = sample
                    .SubtractTensors(predOriginalSample)
                    .DivideTensorByFloat(sigmaInterpol);
                sample = _sample;
                _sample = null;
            }

            _stepIndex += 1;
            return new SchedulerStepResult(sample.AddTensors(derivative.MultiplyTensorByFloat(dt)));
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
            var sigma = _sigmas[_stepIndex];
            return noise
                .MultiplyTensorByFloat(sigma)
                .AddTensors(originalSamples);
        }


        /// <summary>
        /// Interpolates the sigmas.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <returns></returns>
        public float[] InterpolateSigmas(float[] sigmas)
        {
            var rolledLogSigmas = sigmas
                .Append(0f)
                .Select((value, index) => (float)Math.Log(sigmas[(index + sigmas.Length - 1) % sigmas.Length]))
                .ToArray();

            var lerpSigmas = new float[rolledLogSigmas.Length - 1];
            for (int i = 0; i < rolledLogSigmas.Length - 1; i++)
            {
                lerpSigmas[i] = (float)Math.Exp(rolledLogSigmas[i] + 0.5f * (rolledLogSigmas[i + 1] - rolledLogSigmas[i]));
            }
            return lerpSigmas;
        }


        /// <summary>
        /// Interleaves the specified sigmas.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <returns></returns>
        private float[] Interleave(float[] sigmas)
        {
            var first = sigmas.First();
            var last = sigmas.Last();
            return sigmas.Skip(1)
                .SelectMany(value => new[] { value, value })
                .Prepend(first)
                .Append(last)
                .ToArray();
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected override void Dispose(bool disposing)
        {
            _alphasCumProd = null;
            base.Dispose(disposing);
        }
    }
}

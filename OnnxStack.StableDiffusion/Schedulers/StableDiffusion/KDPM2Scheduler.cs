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
    public sealed class KDPM2Scheduler : SchedulerBase
    {
        private int _stepIndex;
        private float[] _sigmasInterpol;
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
            base.Initialize();
            _stepIndex = 0;
            _sample = null;
            Options.TimestepSpacing = TimestepSpacingType.Trailing;
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            // Create timesteps based on the specified strategy
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

            sigmas = [.. sigmas, 0f];
            var sigmasInterpol = InterpolateSigmas(sigmas);

            Sigmas = RepeatInterleave(sigmas);
            _sigmasInterpol = RepeatInterleave(sigmasInterpol);

            SetInitNoiseSigma();

            var timestepsInterpol = SigmaToTimestep(sigmasInterpol, logSigmas);
            var timestepResult = InterpolateTimesteps(timestepsInterpol, timesteps);
            return timestepResult;
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
                ? Sigmas[_stepIndex]
                : _sigmasInterpol[_stepIndex];

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
        /// <exception cref="ArgumentException">Invalid prediction_type: {SchedulerOptions.PredictionType}</exception>
        /// <exception cref="NotImplementedException">KDPM2Scheduler Thresholding currently not implemented</exception>
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int contextSize = 16)
        {
            float sigma;
            float sigmaInterpol;
            float sigmaNext;
            bool isFirstPass = _sample is null;
            if (isFirstPass)
            {
                sigma = Sigmas[_stepIndex];
                sigmaInterpol = _sigmasInterpol[_stepIndex + 1];
                sigmaNext = Sigmas[_stepIndex + 1];
            }
            else
            {
                sigma = Sigmas[_stepIndex - 1];
                sigmaInterpol = _sigmasInterpol[_stepIndex];
                sigmaNext = Sigmas[_stepIndex];
            }

            //# currently only gamma=0 is supported. This usually works best anyways.
            float gamma = 0f;
            float sigmaHat = sigma * (gamma + 1f);
            var sigmaInput = isFirstPass ? sigmaHat : sigmaInterpol;
            var predOriginalSample = GetPredictedSample(modelOutput, sample, sigmaInput);

            DenseTensor<float> sampleResult;
            if (isFirstPass)
            {
                var derivative = sample
                    .SubtractTensors(predOriginalSample)
                    .DivideTensorByFloat(sigmaHat);

                var delta = sigmaInterpol - sigmaHat;
                sampleResult = sample.AddTensors(derivative.MultiplyTensorByFloat(delta));

                _sample = sample.CloneTensor();
            }
            else
            {
                var derivative = sample
                    .SubtractTensors(predOriginalSample)
                    .DivideTensorByFloat(sigmaInterpol);

                var delta = sigmaNext - sigmaHat;
                sampleResult = _sample.AddTensors(derivative.MultiplyTensorByFloat(delta));

                _sample = null;
            }

            _stepIndex += 1;
            return new SchedulerStepResult(sampleResult);
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
            var sigma = Sigmas[_stepIndex];
            return noise
                .MultiplyTensorByFloat(sigma)
                .AddTensors(originalSamples);
        }


        /// <summary>
        /// Repeats the interleave.
        /// </summary>
        /// <param name="input">The input.</param>
        public float[] RepeatInterleave(float[] input)
        {
            int index = 0;
            int resultLength = 1 + (input.Length - 1) * 2 + 1;
            float[] result = new float[resultLength];
            result[index++] = input[0].ZeroIfNan();
            for (int i = 1; i < input.Length; i++)
            {
                result[index++] = input[i].ZeroIfNan();
                result[index++] = input[i].ZeroIfNan();
            }
            result[index] = input[^1].ZeroIfNan();
            return result;
        }


        /// <summary>
        /// Interpolates the sigmas.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        public float[] InterpolateSigmas(float[] sigmas)
        {
            var logSigmas = new float[sigmas.Length];
            var rolledLogSigmas = new float[sigmas.Length];
            var result = new float[sigmas.Length];

            for (int i = 0; i < sigmas.Length; i++)
                logSigmas[i] = MathF.Log(sigmas[i]);

            rolledLogSigmas[0] = logSigmas[sigmas.Length - 1];
            for (int i = 1; i < sigmas.Length; i++)
                rolledLogSigmas[i] = logSigmas[i - 1];

            for (int i = 0; i < sigmas.Length; i++)
            {
                float lerp = logSigmas[i] + 0.5f * (rolledLogSigmas[i] - logSigmas[i]);
                result[i] = MathF.Exp(lerp);
            }
            return result;
        }


        /// <summary>
        /// Interpolates the timesteps.
        /// </summary>
        /// <param name="timestepsInterpol">The timesteps interpol.</param>
        /// <param name="timesteps">The timesteps.</param>
        private int[] InterpolateTimesteps(float[] timestepsInterpol, float[] timesteps)
        {
            var sliceTimesteps = timesteps[1..];
            var sliceInterpol = timestepsInterpol[1..^1];
            var interleaved = new List<int>();
            for (int i = 0; i < sliceTimesteps.Length; i++)
            {
                interleaved.Add((int)Math.Round(sliceInterpol[i]));
                interleaved.Add((int)Math.Round(sliceTimesteps[i]));
            }

            interleaved.Add((int)timesteps[0]);
            return interleaved
                .OrderByDescending(x => x)
                .ToArray();
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected override void Dispose(bool disposing)
        {
            _sigmasInterpol = null;
            base.Dispose(disposing);
        }
    }
}

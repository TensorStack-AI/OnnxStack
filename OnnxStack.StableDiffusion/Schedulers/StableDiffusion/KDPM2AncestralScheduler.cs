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
    public sealed class KDPM2AncestralScheduler : SchedulerBase
    {
        private int _stepIndex;
        private float[] _sigmasInterpol;
        private DenseTensor<float> _previousSample;
        private float[] _sigmas_up;
        private float[] _sigmas_down;

        /// <summary>
        /// Initializes a new instance of the <see cref="KDPM2AncestralScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public KDPM2AncestralScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="KDPM2AncestralScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public KDPM2AncestralScheduler(SchedulerOptions options) : base(options) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            base.Initialize();
            _stepIndex = 0;
            _previousSample = null;
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

            //compute up and down sigmas
            var sigmas_next = RollLeft(sigmas);
            sigmas_next[^1] = 0.0f;
            var sigmas_up = ComputeSigmasUp(sigmas, sigmas_next);
            var sigmas_down = ComputeSigmasDown(sigmas_next, sigmas_up);
            sigmas_down[^1] = 0.0f;

            //compute interpolated sigmas
            var sigmas_interpol = InterpolateSigmas(sigmas, sigmas_down);

            Sigmas = RepeatInterleave(sigmas);
            _sigmasInterpol = RepeatInterleave(sigmas_interpol);
            _sigmas_up = RepeatInterleave(sigmas_up);
            _sigmas_down = RepeatInterleave(sigmas_down);

            SetInitNoiseSigma();

            var timestepsInterpol = SigmaToTimestep(sigmas_interpol, logSigmas)
                .OrderByDescending(x => x)
                .ToArray();
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
            var sigma = _previousSample is null
                ? Sigmas[_stepIndex]
                : _sigmasInterpol[_stepIndex - 1];

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
        /// <exception cref="NotImplementedException">KDPM2AncestralScheduler Thresholding currently not implemented</exception>
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int contextSize = 16)
        {
            float sigma;
            float sigmaInterpol;
            float sigmaUp;
            float sigmaDown = 0f;
            bool isFirstPass = _previousSample is null;
            if (isFirstPass)
            {
                sigma = Sigmas[_stepIndex];
                sigmaInterpol = _sigmasInterpol[_stepIndex];
                sigmaUp = _sigmas_up[_stepIndex];
                if (_stepIndex > 0)
                    sigmaDown = _sigmas_down[_stepIndex - 1];
            }
            else
            {
                sigma = Sigmas[_stepIndex - 1];
                sigmaInterpol = _sigmasInterpol[_stepIndex - 1];
                sigmaUp = _sigmas_up[_stepIndex - 1];
                sigmaDown = _sigmas_down[_stepIndex - 1];
            }

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
                sampleResult = sample.AddTensors(derivative.MultiplyBy(delta));

                _previousSample = sample.CloneTensor();
            }
            else
            {
                var derivative = sample
                    .SubtractTensors(predOriginalSample)
                    .DivideTensorByFloat(sigmaInterpol);

                var delta = sigmaDown - sigmaHat;
                var noise = CreateRandomSample(_previousSample.Dimensions);
                sampleResult = _previousSample
                    .AddTensors(derivative.MultiplyBy(delta))
                    .AddTensors(noise.MultiplyBy(sigmaUp));

                _previousSample = null;
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
        /// Computes the sigmas up.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <param name="sigmasNext">The sigmas next.</param>
        /// <returns>System.Single[].</returns>
        private float[] ComputeSigmasUp(float[] sigmas, float[] sigmasNext)
        {
            var sigmasUp = new float[sigmas.Length];
            for (int i = 0; i < sigmas.Length; i++)
            {
                float sigmaSq = (sigmas[i] * sigmas[i]).ZeroIfNan();
                float sigmaNextSq = (sigmasNext[i] * sigmasNext[i]).ZeroIfNan();
                float value = sigmaNextSq * (sigmaSq - sigmaNextSq) / sigmaSq;
                sigmasUp[i] = MathF.Sqrt(value);
            }
            return sigmasUp;
        }


        /// <summary>
        /// Computes the sigmas down.
        /// </summary>
        /// <param name="sigmasNext">The sigmas next.</param>
        /// <param name="sigmasUp">The sigmas up.</param>
        /// <returns>System.Single[].</returns>
        private float[] ComputeSigmasDown(float[] sigmasNext, float[] sigmasUp)
        {
            var sigmasDown = new float[sigmasNext.Length];
            for (int i = 0; i < sigmasNext.Length; i++)
            {
                float value = (sigmasNext[i] * sigmasNext[i]).ZeroIfNan() - (sigmasUp[i] * sigmasUp[i]).ZeroIfNan();
                sigmasDown[i] = MathF.Sqrt(value);
            }

            return sigmasDown;
        }


        /// <summary>
        /// Interpolates the sigmas.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <param name="sigmasDown">The sigmas down.</param>
        /// <returns>System.Single[].</returns>
        private float[] InterpolateSigmas(float[] sigmas, float[] sigmasDown)
        {
            var sigmasInterpol = new float[sigmas.Length];
            for (int i = 0; i < sigmas.Length; i++)
            {
                float logSigma = MathF.Log(sigmas[i]).ZeroIfNan();
                float logSigmaDown = MathF.Log(sigmasDown[i]).ZeroIfNan();

                float lerp = float.Lerp(logSigma, logSigmaDown, 0.5f);
                sigmasInterpol[i] = MathF.Exp(lerp);
            }

            if (sigmas.Length >= 2)
            {
                sigmasInterpol[^2] = 0f;
                sigmasInterpol[^1] = 0f;
            }
            return sigmasInterpol;
        }


        /// <summary>
        /// Interpolates the timesteps.
        /// </summary>
        /// <param name="timestepsInterpol">The timesteps interpol.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns>System.Int32[].</returns>
        private int[] InterpolateTimesteps(float[] timestepsInterpol, float[] timesteps)
        {
            var sliceTimesteps = timesteps[1..];
            var sliceInterpol = timestepsInterpol[..^2];
            var interleaved = new List<int>();
            for (int i = 0; i < sliceInterpol.Length; i++)
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
        /// Rolls array to the left.
        /// </summary>
        /// <param name="sigmas">The sigmas.</param>
        /// <returns>System.Single[].</returns>
        private float[] RollLeft(float[] sigmas)
        {
            var result = new float[sigmas.Length];
            for (int i = 0; i < sigmas.Length - 1; i++)
                result[i] = sigmas[i + 1];

            result[^1] = sigmas[0];
            return result;
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected override void Dispose(bool disposing)
        {
            _sigmas_up = null;
            _sigmas_down = null;
            _sigmasInterpol = null;
            _previousSample = null;
            base.Dispose(disposing);
        }
    }
}

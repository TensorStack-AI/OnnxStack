using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    internal class DDPMWuerstchenScheduler : SchedulerBase
    {
        private float _s;
        private float _scaler;
        private float _initAlphaCumprod;


        /// <summary>
        /// Initializes a new instance of the <see cref="DDPMWuerstchenScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public DDPMWuerstchenScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="DDPMWuerstchenScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public DDPMWuerstchenScheduler(SchedulerOptions options) : base(options) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            _s = 0.008f;
            _scaler = 1.0f;
            _initAlphaCumprod = MathF.Pow(MathF.Cos(_s / (1f + _s) * MathF.PI * 0.5f), 2f);
            SetInitNoiseSigma(1.0f);
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            // Create timesteps based on the specified strategy
            var timesteps = ArrayHelpers.Linspace(0, 1000, Options.InferenceSteps + 1);
            var x = timesteps
                .Skip(1)
                .Select(x => (int)x)
                .OrderByDescending(x => x)
                .ToArray();
            return x;
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="sample">The sample.</param>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public override DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
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
        /// <exception cref="ArgumentException">Invalid prediction_type: {SchedulerOptions.PredictionType}</exception>
        /// <exception cref="NotImplementedException">DDPMScheduler Thresholding currently not implemented</exception>
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            var currentTimestep = timestep / 1000f;
            var previousTimestep = GetPreviousTimestep(timestep) / 1000f;

            var alpha_cumprod = GetAlphaCumprod(currentTimestep);
            var alpha_cumprod_prev = GetAlphaCumprod(previousTimestep);
            var alpha = alpha_cumprod / alpha_cumprod_prev;

            var predictedSample = sample
                .SubtractTensors(modelOutput.MultiplyTensorByFloat(1f - alpha).DivideTensorByFloat(MathF.Sqrt(1f - alpha_cumprod)))
                .MultiplyTensorByFloat(MathF.Sqrt(1f / alpha))
                .AddTensors(CreateRandomSample(modelOutput.Dimensions)
                .MultiplyTensorByFloat(MathF.Sqrt((1f - alpha) * (1f - alpha_cumprod_prev) / (1f - alpha_cumprod))));

            return new SchedulerStepResult(predictedSample);
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
            float timestep = timesteps[0] / 1000f;
            float alphaProd = GetAlphaCumprod(timestep);
            float sqrtAlpha = MathF.Sqrt(alphaProd);
            float sqrtOneMinusAlpha = MathF.Sqrt(1.0f - alphaProd);

            return noise
                .MultiplyTensorByFloat(sqrtOneMinusAlpha)
                .AddTensors(originalSamples.MultiplyTensorByFloat(sqrtAlpha));
        }


        /// <summary>
        /// Gets the previous timestep.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        protected override int GetPreviousTimestep(int timestep)
        {
            var index = Timesteps.IndexOf(timestep) + 1;
            if (index > Timesteps.Count - 1)
                return 0;

            return Timesteps[index];
        }


        private float GetAlphaCumprod(float timestep)
        {
            if (_scaler > 1.0f)
                timestep = 1f - MathF.Pow(1f - timestep, _scaler);
            else if (_scaler < 1.0f)
                timestep = MathF.Pow(timestep, _scaler);

            var alphaCumprod = MathF.Pow(MathF.Cos((timestep + _s) / (1f + _s) * MathF.PI * 0.5f), 2f) / _initAlphaCumprod;
            return Math.Clamp(alphaCumprod, 0.0001f, 0.9999f);
        }


        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
        }
    }
}

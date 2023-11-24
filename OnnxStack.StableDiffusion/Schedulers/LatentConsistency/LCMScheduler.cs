using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;

namespace OnnxStack.StableDiffusion.Schedulers.LatentConsistency
{
    internal class LCMScheduler : SchedulerBase
    {
        private float[] _alphasCumProd;
        private float _finalAlphaCumprod;

        /// <summary>
        /// Initializes a new instance of the <see cref="LCMScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public LCMScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="LCMScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public LCMScheduler(SchedulerOptions options) : base(options) { }


        /// <summary>
        /// Gets the compatible pipeline.
        /// </summary>
        public override DiffuserPipelineType PipelineType => DiffuserPipelineType.LatentConsistency;


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            _alphasCumProd = null;

            var betas = GetBetaSchedule();
            var alphas = betas.Select(beta => 1.0f - beta);
            _alphasCumProd = alphas
                .Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b))
                .ToArray();

            bool setAlphaToOne = true;
            _finalAlphaCumprod = setAlphaToOne
                ? 1.0f
            : _alphasCumProd.First();

            SetInitNoiseSigma(1.0f);
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            // LCM Timesteps Setting
            // Currently, only linear spacing is supported.
            var timeIncrement = Options.TrainTimesteps / Options.OriginalInferenceSteps;

            //# LCM Training Steps Schedule
            var lcmOriginTimesteps = Enumerable.Range(1, Options.OriginalInferenceSteps)
                .Select(x => x * timeIncrement - 1)
                .ToArray();

            var steps = ArrayHelpers.Linspace(0, lcmOriginTimesteps.Length - 1, Options.InferenceSteps)
               .Select(x => (int)Math.Floor(x))
               .Select(x => lcmOriginTimesteps[x])
               .OrderByDescending(x => x)
               .ToArray();
             return steps;
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
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            //# Latent Consistency Models paper https://arxiv.org/abs/2310.04378

            int currentTimestep = timestep;

            // 1. get previous step value
            int prevIndex = Timesteps.IndexOf(currentTimestep) + 1;
            int previousTimestep = prevIndex < Timesteps.Count
                ? Timesteps[prevIndex]
                : currentTimestep;

            //# 2. compute alphas, betas
            float alphaProdT = _alphasCumProd[currentTimestep];
            float alphaProdTPrev = previousTimestep >= 0
                ? _alphasCumProd[previousTimestep]
                : _finalAlphaCumprod;
            float betaProdT = 1f - alphaProdT;
            float betaProdTPrev = 1f - alphaProdTPrev;

            float alphaProdTSqrt = MathF.Sqrt(alphaProdT);
            float betaProdTSqrt = MathF.Sqrt(betaProdT);
            float betaProdTPrevSqrt = MathF.Sqrt(betaProdTPrev);
            float alphaProdTPrevSqrt = MathF.Sqrt(alphaProdTPrev);


            // 3.Get scalings for boundary conditions
            (float cSkip, float cOut) = GetBoundaryConditionScalings(currentTimestep);


            //# 4. compute predicted original sample from predicted noise also called "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            DenseTensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                predOriginalSample = sample
                    .SubtractTensors(modelOutput.MultiplyTensorByFloat(betaProdTSqrt))
                    .DivideTensorByFloat(alphaProdTSqrt);
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                predOriginalSample = modelOutput;
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                predOriginalSample = sample
                    .MultiplyTensorByFloat(alphaProdTSqrt)
                    .SubtractTensors(modelOutput.MultiplyTensorByFloat(betaProdTSqrt));
            }


            //# 5. Clip or threshold "predicted x_0"
            // TODO: OnnxStack does not yet support Threshold and Clipping


            //# 6. Denoise model output using boundary conditions
            var denoised = sample
                .MultiplyTensorByFloat(cSkip)
                .AddTensors(predOriginalSample.MultiplyTensorByFloat(cOut));


            //# 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
            //# Noise is not used on the final timestep of the timestep schedule.
            //# This also means that noise is not used for one-step sampling.
            if (Timesteps.IndexOf(currentTimestep) != Options.InferenceSteps - 1)
            {
                var noise = CreateRandomSample(modelOutput.Dimensions);
                predOriginalSample = noise
                    .MultiplyTensorByFloat(betaProdTPrevSqrt)
                    .AddTensors(denoised.MultiplyTensorByFloat(alphaProdTPrevSqrt));
            }
            else
            {
                predOriginalSample = denoised;
            }


            return new SchedulerStepResult(predOriginalSample, denoised);
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
            // Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L456
            int timestep = timesteps[0];
            float alphaProd = _alphasCumProd[timestep];
            float sqrtAlpha = MathF.Sqrt(alphaProd);
            float sqrtOneMinusAlpha = MathF.Sqrt(1.0f - alphaProd);

            return noise
                .MultiplyTensorByFloat(sqrtOneMinusAlpha)
                .AddTensors(originalSamples.MultiplyTensorByFloat(sqrtAlpha));
        }


        /// <summary>
        /// Gets the boundary condition scalings.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <returns></returns>
        public (float cSkip, float cOut) GetBoundaryConditionScalings(float timestep)
        {
            //self.sigma_data = 0.5  # Default: 0.5
            var sigmaData = 0.5f;
            var timestepScaling = 10f;
            var scaledTimestep = timestepScaling * timestep;

            float c = MathF.Pow(scaledTimestep, 2f) + MathF.Pow(sigmaData, 2f);
            float cSkip = MathF.Pow(sigmaData, 2f) / c;
            float cOut = scaledTimestep / MathF.Pow(c, 0.5f);
            return (cSkip, cOut);
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

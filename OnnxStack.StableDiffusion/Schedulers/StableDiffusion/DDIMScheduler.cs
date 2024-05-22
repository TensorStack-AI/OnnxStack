using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    internal class DDIMScheduler : SchedulerBase
    {
        private float[] _alphasCumProd;
        private float _finalAlphaCumprod;

        /// <summary>
        /// Initializes a new instance of the <see cref="DDIMScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public DDIMScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="DDIMScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public DDIMScheduler(SchedulerOptions options) : base(options) { }


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
            // Create timesteps based on the specified strategy
            var timesteps = GetTimesteps();
            return timesteps
                .Select(x => (int)x)
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
        /// <exception cref="NotImplementedException">DDIMScheduler Thresholding currently not implemented</exception>
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            //# See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
            //# Ideally, read DDIM paper in-detail understanding

            //# Notation (<variable name> -> <name in paper>
            //# - pred_noise_t -> e_theta(x_t, t)
            //# - pred_original_sample -> f_theta(x_t, t) or x_0
            //# - std_dev_t -> sigma_t
            //# - eta -> η
            //# - pred_sample_direction -> "direction pointing to x_t"
            //# - pred_prev_sample -> "x_t-1"

            int currentTimestep = timestep;
            int previousTimestep = GetPreviousTimestep(currentTimestep);

            //# 1. compute alphas, betas
            float alphaProdT = _alphasCumProd[currentTimestep];
            float alphaProdTPrev = previousTimestep >= 0 ? _alphasCumProd[previousTimestep] : _finalAlphaCumprod;
            float betaProdT = 1f - alphaProdT;


            //# 2. compute predicted original sample from predicted noise also called
            //# "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            DenseTensor<float> predEpsilon = null;
            DenseTensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                var sampleBeta = sample.SubtractTensors(modelOutput.MultiplyTensorByFloat(MathF.Sqrt(betaProdT)));
                predOriginalSample = sampleBeta.DivideTensorByFloat(MathF.Sqrt(alphaProdT));
                predEpsilon = modelOutput;
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                predOriginalSample = modelOutput;
                predEpsilon = sample.SubtractTensors(predOriginalSample
                    .MultiplyTensorByFloat(MathF.Sqrt(alphaProdT)))
                    .DivideTensorByFloat(MathF.Sqrt(betaProdT));
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                var tmp = sample.MultiplyTensorByFloat(MathF.Pow(alphaProdT, 0.5f));
                var tmp2 = modelOutput.MultiplyTensorByFloat(MathF.Pow(betaProdT, 0.5f));
                predOriginalSample = tmp.Subtract(tmp2);

                var tmp3 = modelOutput.MultiplyTensorByFloat(MathF.Pow(alphaProdT, 0.5f));
                var tmp4 = sample.MultiplyTensorByFloat(MathF.Pow(betaProdT, 0.5f));
                predEpsilon = tmp3.Add(tmp4);
            }


            //# 3. Clip or threshold "predicted x_0"
            if (Options.Thresholding)
            {
                // TODO:
                // predOriginalSample = ThresholdSample(predOriginalSample);
            }
            else if (Options.ClipSample)
            {
                predOriginalSample = predOriginalSample.Clip(-Options.ClipSampleRange, Options.ClipSampleRange);
            }


            //# 4. compute variance: "sigma_t(η)" -> see formula (16)
            //# σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            var eta = 0f;
            var variance = GetVariance(currentTimestep, previousTimestep);
            var stdDevT = eta * MathF.Pow(variance, 0.5f);

            var useClippedModelOutput = false;
            if (useClippedModelOutput)
            {
                //# the pred_epsilon is always re-derived from the clipped x_0 in Glide
                predEpsilon = sample
                    .SubtractTensors(predOriginalSample.MultiplyTensorByFloat(MathF.Pow(alphaProdT, 0.5f)))
                    .DivideTensorByFloat(MathF.Pow(betaProdT, 0.5f));
            }


            //# 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            var predSampleDirection = predEpsilon.MultiplyTensorByFloat(MathF.Pow(1.0f - alphaProdTPrev - MathF.Pow(stdDevT, 2f), 0.5f));

            //# 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            var prevSample = predSampleDirection.AddTensors(predOriginalSample.MultiplyTensorByFloat(MathF.Pow(alphaProdTPrev, 0.5f)));

            if (eta > 0)
                prevSample = prevSample.AddTensors(CreateRandomSample(modelOutput.Dimensions).MultiplyTensorByFloat(stdDevT));

            return new SchedulerStepResult(prevSample);
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
        /// Gets the variance.
        /// </summary>
        /// <param name="timestep">The t.</param>
        /// <param name="predictedVariance">The predicted variance.</param>
        /// <returns></returns>
        private float GetVariance(int timestep, int prevTimestep)
        {
            float alphaProdT = _alphasCumProd[timestep];
            float alphaProdTPrev = prevTimestep >= 0
                ? _alphasCumProd[timestep]
                : _finalAlphaCumprod;

            float betaProdT = 1f - alphaProdT;
            float betaProdTPrev = 1f - alphaProdTPrev;
            float variance = betaProdTPrev / betaProdT * (1f - alphaProdT / alphaProdTPrev);
            return variance;
        }


        protected override void Dispose(bool disposing)
        {
            _alphasCumProd = null;
            base.Dispose(disposing);
        }
    }
}

using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    public sealed class DDPMScheduler : SchedulerBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DDPMScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public DDPMScheduler() : this(new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="DDPMScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public DDPMScheduler(SchedulerOptions options) : base(options) { }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            base.Initialize();
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            var timesteps = GetTimesteps();
            return timesteps
                .Select(x => (int)Math.Round(x))
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
        /// <exception cref="NotImplementedException">DDPMScheduler Thresholding currently not implemented</exception>
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int contextSize = 16)
        {
            int currentTimestep = timestep;
            int currentTimestepIndex = Timesteps.IndexOf(currentTimestep);
            int previousTimestepIndex = currentTimestepIndex + 1;
            int previousTimestep = Timesteps.ElementAtOrDefault(previousTimestepIndex);

            //# 1. compute alphas, betas
            float alphaProdT = AlphasCumProd[currentTimestep];
            float alphaProdTPrev = previousTimestep >= 0 ? AlphasCumProd[previousTimestep] : 1f;
            float betaProdT = 1f - alphaProdT;
            float betaProdTPrev = 1f - alphaProdTPrev;
            float currentAlphaT = alphaProdT / alphaProdTPrev;
            float currentBetaT = 1f - currentAlphaT;

            float predictedVariance = 0;


            // TODO: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L390
            //if (modelOutput.Dimensions[1] == sample.Dimensions[1] * 2 && VarianceTypeIsLearned())
            //{
            //    DenseTensor<float>[] splitModelOutput = modelOutput.Split(modelOutput.Dimensions[1] / 2, 1);
            //    TensorHelper.SplitTensor(modelOutput, )
            //    modelOutput = splitModelOutput[0];
            //    predictedVariance = splitModelOutput[1];
            //}


            //# 2. compute predicted original sample from predicted noise also called
            //# "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            var predOriginalSample = GetPredictedSample(modelOutput, sample, alphaProdT, betaProdT);

            //# 3. Clip or threshold "predicted x_0"
            if (Options.Thresholding)
            {
                // TODO: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L322
                // predOriginalSample = ThresholdSample(predOriginalSample);
            }
            else if (Options.ClipSample)
            {
                predOriginalSample = predOriginalSample.Clip(-Options.ClipSampleRange, Options.ClipSampleRange);
            }

            //# 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            //# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            float predOriginalSampleCoeff = (float)Math.Sqrt(alphaProdTPrev) * currentBetaT / betaProdT;
            float currentSampleCoeff = (float)Math.Sqrt(currentAlphaT) * betaProdTPrev / betaProdT;


            //# 5. Compute predicted previous sample µ_t
            //# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            //pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
            var predPrevSample = sample
                .MultiplyTensorByFloat(currentSampleCoeff)
                .AddTensors(predOriginalSample.MultiplyTensorByFloat(predOriginalSampleCoeff));


            //# 6. Add noise
            if (currentTimestep > 0)
            {
                DenseTensor<float> variance;
                var varianceNoise = CreateRandomSample(modelOutput.Dimensions);
                if (Options.VarianceType == VarianceType.FixedSmallLog)
                {
                    var v = GetVariance(currentTimestep, predictedVariance);
                    variance = varianceNoise.MultiplyTensorByFloat(v);
                }
                else if (Options.VarianceType == VarianceType.LearnedRange)
                {
                    var v = (float)Math.Exp(0.5 * GetVariance(currentTimestep, predictedVariance));
                    variance = varianceNoise.MultiplyTensorByFloat(v);
                }
                else
                {
                    var v = (float)Math.Sqrt(GetVariance(currentTimestep, predictedVariance));
                    variance = varianceNoise.MultiplyTensorByFloat(v);
                }
                predPrevSample = predPrevSample.AddTensors(variance);
            }

            return new SchedulerStepResult(predPrevSample);
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
            float alphaProd = AlphasCumProd[timestep];
            float sqrtAlpha = (float)Math.Sqrt(alphaProd);
            float sqrtOneMinusAlpha = (float)Math.Sqrt(1.0f - alphaProd);

            return noise
                .MultiplyTensorByFloat(sqrtOneMinusAlpha)
                .AddTensors(originalSamples.MultiplyTensorByFloat(sqrtAlpha));
        }


        /// <summary>
        /// Gets the predicted sample.
        /// </summary>
        /// <param name="modelOutput">The model output.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="alphaProdT">The alpha product t.</param>
        /// <param name="betaProdT">The beta product t.</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        private DenseTensor<float> GetPredictedSample(DenseTensor<float> modelOutput, DenseTensor<float> sample, float alphaProdT, float betaProdT)
        {
            DenseTensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                //pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                var sampleBeta = sample.SubtractTensors(modelOutput.MultiplyTensorByFloat((float)Math.Sqrt(betaProdT)));
                predOriginalSample = sampleBeta.DivideTensorByFloat((float)Math.Sqrt(alphaProdT));
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                predOriginalSample = modelOutput;
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                // pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
                var alphaSqrt = (float)Math.Sqrt(alphaProdT);
                var betaSqrt = (float)Math.Sqrt(betaProdT);
                predOriginalSample = sample
                    .MultiplyTensorByFloat(alphaSqrt)
                    .SubtractTensors(modelOutput.MultiplyTensorByFloat(betaSqrt));
            }
            return predOriginalSample;
        }


        /// <summary>
        /// Gets the variance.
        /// </summary>
        /// <param name="timestep">The t.</param>
        /// <param name="predictedVariance">The predicted variance.</param>
        /// <returns></returns>
        private float GetVariance(int timestep, float predictedVariance = 0f)
        {
            int prevTimestep = GetPreviousTimestep(timestep);
            float alphaProdT = AlphasCumProd[timestep];
            float alphaProdTPrev = prevTimestep >= 0 ? AlphasCumProd[prevTimestep] : 1.0f;
            float currentBetaT = 1 - alphaProdT / alphaProdTPrev;

            // For t > 0, compute predicted variance βt
            float variance = (1 - alphaProdTPrev) / (1 - alphaProdT) * currentBetaT;

            // Clamp variance to ensure it's not 0
            variance = Math.Max(variance, 1e-20f);


            if (Options.VarianceType == VarianceType.FixedSmallLog)
            {
                variance = (float)Math.Exp(0.5 * Math.Log(variance));
            }
            else if (Options.VarianceType == VarianceType.FixedLarge)
            {
                variance = currentBetaT;
            }
            else if (Options.VarianceType == VarianceType.FixedLargeLog)
            {
                variance = (float)Math.Log(currentBetaT);
            }
            else if (Options.VarianceType == VarianceType.Learned)
            {
                return predictedVariance;
            }
            else if (Options.VarianceType == VarianceType.LearnedRange)
            {
                float minLog = (float)Math.Log(variance);
                float maxLog = (float)Math.Log(currentBetaT);
                float frac = (predictedVariance + 1) / 2;
                variance = frac * maxLog + (1 - frac) * minLog;
            }
            return variance;
        }

    }
}

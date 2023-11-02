using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers.StableDiffusion
{
    internal class DDPMScheduler : SchedulerBase
    {
        private float[] _alphasCumProd;

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
            _alphasCumProd = null;

            var betas = GetBetaSchedule();
            var alphas = betas.Select(beta => 1.0f - beta);
            _alphasCumProd = alphas
                .Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b))
                .ToArray();

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
        /// <exception cref="NotImplementedException">DDPMScheduler Thresholding currently not implemented</exception>
        public override SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            int currentTimestep = timestep;
            int previousTimestep = GetPreviousTimestep(currentTimestep);

            //# 1. compute alphas, betas
            float alphaProdT = _alphasCumProd[currentTimestep];
            float alphaProdTPrev = previousTimestep >= 0 ? _alphasCumProd[previousTimestep] : 1f;
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
                predOriginalSample = ThresholdSample(predOriginalSample);
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
                .MultipleTensorByFloat(currentSampleCoeff)
                .AddTensors(predOriginalSample.MultipleTensorByFloat(predOriginalSampleCoeff));


            //# 6. Add noise
            if (currentTimestep > 0)
            {
                DenseTensor<float> variance;
                var varianceNoise = CreateRandomSample(modelOutput.Dimensions);
                if (Options.VarianceType == VarianceType.FixedSmallLog)
                {
                    var v = GetVariance(currentTimestep, predictedVariance);
                    variance = varianceNoise.MultipleTensorByFloat(v);
                }
                else if (Options.VarianceType == VarianceType.LearnedRange)
                {
                    var v = (float)Math.Exp(0.5 * GetVariance(currentTimestep, predictedVariance));
                    variance = varianceNoise.MultipleTensorByFloat(v);
                }
                else
                {
                    var v = (float)Math.Sqrt(GetVariance(currentTimestep, predictedVariance));
                    variance = varianceNoise.MultipleTensorByFloat(v);
                }
                predPrevSample = predPrevSample.AddTensors(variance);
            }

            return new SchedulerStepResult(predPrevSample);
        }


        private DenseTensor<float> GetPredictedSample(DenseTensor<float> modelOutput, DenseTensor<float> sample, float alphaProdT, float betaProdT)
        {
            DenseTensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                //pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                var sampleBeta = sample.SubtractTensors(modelOutput.MultipleTensorByFloat((float)Math.Sqrt(betaProdT)));
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
                    .MultipleTensorByFloat(alphaSqrt)
                    .SubtractTensors(modelOutput.MultipleTensorByFloat(betaSqrt));
            }
            return predOriginalSample;
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
            float sqrtAlpha = (float)Math.Sqrt(alphaProd);
            float sqrtOneMinusAlpha = (float)Math.Sqrt(1.0f - alphaProd);

            return noise
                .MultipleTensorByFloat(sqrtOneMinusAlpha)
                .AddTensors(originalSamples.MultipleTensorByFloat(sqrtAlpha));
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
            float alphaProdT = _alphasCumProd[timestep];
            float alphaProdTPrev = prevTimestep >= 0 ? _alphasCumProd[prevTimestep] : 1.0f;
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


        /// <summary>
        /// Thresholds the sample.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="dynamicThresholdingRatio">The dynamic thresholding ratio.</param>
        /// <param name="sampleMaxValue">The sample maximum value.</param>
        /// <returns></returns>
        private DenseTensor<float> ThresholdSample(DenseTensor<float> input, float dynamicThresholdingRatio = 0.995f, float sampleMaxValue = 1f)
        {
            var sample = new NDArray(input.ToArray(), new Shape(input.Dimensions.ToArray()));
            var batch_size = sample.shape[0];
            var channels = sample.shape[1];
            var height = sample.shape[2];
            var width = sample.shape[3];

            // Flatten sample for doing quantile calculation along each image
            var flatSample = sample.reshape(batch_size, channels * height * width);

            // Calculate the absolute values of the sample
            var absSample = np.abs(flatSample);

            // Calculate the quantile for each row
            var quantiles = new List<float>();
            for (int i = 0; i < batch_size; i++)
            {
                var row = absSample[$"{i},:"].MakeGeneric<float>();
                var percentileValue = CalculatePercentile(row, dynamicThresholdingRatio);
                percentileValue = Math.Clamp(percentileValue, 1f, sampleMaxValue);
                quantiles.Add(percentileValue);
            }

            // Create an NDArray from quantiles
            var quantileArray = np.array(quantiles.ToArray());

            // Calculate the thresholded sample
            var sExpanded = np.expand_dims(quantileArray, 1); // Expand to match the sample shape
            var negSExpanded = np.negative(sExpanded); // Get the negation of sExpanded
            var thresholdedSample = sample - negSExpanded; // Element-wise subtraction
            thresholdedSample = np.maximum(thresholdedSample, negSExpanded); // Ensure values are not less than -sExpanded
            thresholdedSample = np.minimum(thresholdedSample, sExpanded); // Ensure values are not greater than sExpanded
            thresholdedSample = thresholdedSample / sExpanded;

            // Reshape to the original shape
            thresholdedSample = thresholdedSample.reshape(batch_size, channels, height, width);

            return TensorHelper.CreateTensor(thresholdedSample.ToArray<float>(), thresholdedSample.shape);
        }


        /// <summary>
        /// Calculates the percentile.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="percentile">The percentile.</param>
        /// <returns></returns>
        private float CalculatePercentile(NDArray data, float percentile)
        {
            // Sort the data indices in ascending order
            var sortedIndices = np.argsort<float>(data);

            // Calculate the index corresponding to the percentile
            var index = (int)Math.Ceiling(percentile / 100f * (data.Shape[0] - 1));

            // Retrieve the value at the calculated index
            var percentileValue = data[sortedIndices[index]];

            return percentileValue.GetSingle();
        }


        /// <summary>
        /// Determines whether the VarianceType is learned.
        /// </summary>
        /// <returns>
        ///   <c>true</c> if the VarianceType is learned; otherwise, <c>false</c>.
        /// </returns>
        private bool IsVarianceTypeLearned()
        {
            return Options.VarianceType == VarianceType.Learned || Options.VarianceType == VarianceType.LearnedRange;
        }

        protected override void Dispose(bool disposing)
        {
            _alphasCumProd = null;
            base.Dispose(disposing);
        }
    }
}

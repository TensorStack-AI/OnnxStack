using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxStack.StableDiffusion.Schedulers
{
    internal class DDPMScheduler : SchedulerBase
    {
        private float[] _betas;
        private List<float> _alphasCumulativeProducts;

        /// <summary>
        /// Initializes a new instance of the <see cref="DDPMScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        public DDPMScheduler(StableDiffusionOptions stableDiffusionOptions)
            : this(stableDiffusionOptions, new SchedulerOptions()) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="DDPMScheduler"/> class.
        /// </summary>
        /// <param name="stableDiffusionOptions">The stable diffusion options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        public DDPMScheduler(StableDiffusionOptions stableDiffusionOptions, SchedulerOptions schedulerOptions)
            : base(stableDiffusionOptions, schedulerOptions)
        {
        }


        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected override void Initialize()
        {
            var alphas = new List<float>();
            if (SchedulerOptions.TrainedBetas != null)
            {
                _betas = SchedulerOptions.TrainedBetas.ToArray();
            }
            else if (SchedulerOptions.BetaSchedule == BetaSchedule.Linear)
            {
                _betas = np.linspace(SchedulerOptions.BetaStart, SchedulerOptions.BetaEnd, SchedulerOptions.TrainTimesteps).ToArray<float>();
            }
            else if (SchedulerOptions.BetaSchedule == BetaSchedule.ScaledLinear)
            {
                // This schedule is very specific to the latent diffusion model.
                _betas = np.power(np.linspace(MathF.Sqrt(SchedulerOptions.BetaStart), MathF.Sqrt(SchedulerOptions.BetaEnd), SchedulerOptions.TrainTimesteps), 2).ToArray<float>();
            }
            else if (SchedulerOptions.BetaSchedule == BetaSchedule.SquaredCosCapV2)
            {
                // Glide cosine schedule
                _betas = GetBetasForAlphaBar();
            }
            //else if (betaSchedule == "sigmoid")
            //{
            //    // GeoDiff sigmoid schedule
            //    var betas = np.linspace(-6, 6, numTrainTimesteps);
            //    Betas = (np.multiply(np.exp(betas), (betaEnd - betaStart)) + betaStart).ToArray<float>();
            //}


            for (int i = 0; i < SchedulerOptions.TrainTimesteps; i++)
            {
                alphas.Add(1.0f - _betas[i]);
            }

            _alphasCumulativeProducts = new List<float> { alphas[0] };
            for (int i = 1; i < SchedulerOptions.TrainTimesteps; i++)
            {
                _alphasCumulativeProducts.Add(_alphasCumulativeProducts[i - 1] * alphas[i]);
            }

            SetInitNoiseSigma(1.0f);
        }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            // Create timesteps based on the specified strategy
            NDArray timestepsArray = null;
            if (SchedulerOptions.TimestepSpacing == TimestepSpacing.Linspace)
            {
                timestepsArray = np.linspace(0, SchedulerOptions.TrainTimesteps - 1, StableDiffusionOptions.NumInferenceSteps);
                timestepsArray = np.around(timestepsArray)["::1"];
            }
            else if (SchedulerOptions.TimestepSpacing == TimestepSpacing.Leading)
            {
                var stepRatio = SchedulerOptions.TrainTimesteps / StableDiffusionOptions.NumInferenceSteps;
                timestepsArray = np.arange(0, StableDiffusionOptions.NumInferenceSteps) * stepRatio;
                timestepsArray = np.around(timestepsArray)["::1"];
                timestepsArray += SchedulerOptions.StepsOffset;
            }
            else if (SchedulerOptions.TimestepSpacing == TimestepSpacing.Trailing)
            {
                var stepRatio = SchedulerOptions.TrainTimesteps / StableDiffusionOptions.NumInferenceSteps;
                timestepsArray = np.arange(SchedulerOptions.TrainTimesteps, 0, -stepRatio);
                timestepsArray = np.around(timestepsArray);
                timestepsArray -= 1;
            }

            return  timestepsArray
                .ToArray<float>()
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
        /// <exception cref="System.ArgumentException">Invalid prediction_type: {SchedulerOptions.PredictionType}</exception>
        /// <exception cref="System.NotImplementedException">DDPMScheduler Thresholding currently not implemented</exception>
        public override DenseTensor<float> Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            int currentTimestep = timestep;
            int previousTimestep = GetPreviousTimestep(currentTimestep);

            //# 1. compute alphas, betas
            float alphaProdT = _alphasCumulativeProducts[currentTimestep];
            float alphaProdTPrev = previousTimestep >= 0 ? _alphasCumulativeProducts[previousTimestep] : 1f;
            float betaProdT = 1 - alphaProdT;
            float betaProdTPrev = 1 - alphaProdTPrev;
            float currentAlphaT = alphaProdT / alphaProdTPrev;
            float currentBetaT = 1 - currentAlphaT;

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
            DenseTensor<float> predOriginalSample = null;
            if (SchedulerOptions.PredictionType == PredictionType.Epsilon)
            {
                //pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                var sampleBeta = TensorHelper.SubtractTensors(sample, TensorHelper.MultipleTensorByFloat(modelOutput, (float)Math.Sqrt(betaProdT)));
                predOriginalSample = TensorHelper.DivideTensorByFloat(sampleBeta, (float)Math.Sqrt(alphaProdT), sampleBeta.Dimensions);
            }
            else if (SchedulerOptions.PredictionType == PredictionType.Aample)
            {
                predOriginalSample = modelOutput;
            }
            else if (SchedulerOptions.PredictionType == PredictionType.VariablePrediction)
            {
                // pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
                var alphaSqrt = (float)Math.Sqrt(alphaProdT);
                var betaSqrt = (float)Math.Sqrt(betaProdT);
                predOriginalSample = new DenseTensor<float>((int)sample.Length);
                for (int i = 0; i < sample.Length - 1; i++)
                {
                    predOriginalSample.SetValue(i, alphaSqrt * sample.GetValue(i) - betaSqrt * modelOutput.GetValue(i));
                }
            }


            //# 3. Clip or threshold "predicted x_0"
            if (SchedulerOptions.Thresholding)
            {
                // TODO: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L322
                //predOriginalSample = ThresholdSample(predOriginalSample);
                throw new NotImplementedException("DDPMScheduler Thresholding currently not implemented");
            }
            else if (SchedulerOptions.ClipSample)
            {
                predOriginalSample = TensorHelper.Clip(predOriginalSample, -SchedulerOptions.ClipSampleRange, SchedulerOptions.ClipSampleRange);
            }

            //# 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            //# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            float predOriginalSampleCoeff = ((float)Math.Sqrt(alphaProdTPrev) * currentBetaT) / betaProdT;
            float currentSampleCoeff = (float)Math.Sqrt(currentAlphaT) * betaProdTPrev / betaProdT;


            //# 5. Compute predicted previous sample µ_t
            //# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            //pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
            var pred_sample = TensorHelper.MultipleTensorByFloat(sample, currentSampleCoeff);
            var pred_original = TensorHelper.MultipleTensorByFloat(predOriginalSample, predOriginalSampleCoeff);
            var predPrevSample = TensorHelper.AddTensors(pred_sample, pred_original);


            //# 6. Add noise
            if (currentTimestep > 0)
            {
                DenseTensor<float> variance;
                var varianceNoise = TensorHelper.GetRandomTensor(Random, modelOutput.Dimensions);
                if (SchedulerOptions.VarianceType ==  VarianceType.FixedSmallLog)
                {
                    variance = TensorHelper.MultipleTensorByFloat(varianceNoise, GetVariance(currentTimestep, predictedVariance));
                }
                else if (SchedulerOptions.VarianceType == VarianceType.LearnedRange)
                {
                    var v = (float)Math.Exp(0.5 * GetVariance(currentTimestep, predictedVariance));
                    variance = TensorHelper.MultipleTensorByFloat(varianceNoise, v);
                }
                else
                {
                    var v = (float)Math.Sqrt(GetVariance(currentTimestep, predictedVariance));
                    variance = TensorHelper.MultipleTensorByFloat(varianceNoise, v);
                }
                predPrevSample = TensorHelper.AddTensors(predPrevSample, variance);
            }

            return predPrevSample;
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="originalSamples">The original samples.</param>
        /// <param name="noise">The noise.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        public override DenseTensor<float> AddNoise(DenseTensor<float> originalSamples, DenseTensor<float> noise, int[] timesteps)
        {
            // Make sure alphas_cumprod and timestep have the same device and dtype as originalSamples
            var alphasCumprod = new DenseTensor<float>(_alphasCumulativeProducts.ToArray(), new int[] { _alphasCumulativeProducts.Count });// Convert to DenseTensor
            var timestepsTensor = new DenseTensor<int>(timesteps);

            var sqrtAlphaProd = new DenseTensor<float>(timesteps.Length);
            var sqrtOneMinusAlphaProd = new DenseTensor<float>(timesteps.Length);

            for (int i = 0; i < timesteps.Length; i++)
            {
                int timestep = timesteps[i];
                float alphaProd = alphasCumprod[0, timestep]; // Assuming alphasCumprod is a 2D tensor
                float sqrtAlpha = (float)Math.Sqrt(alphaProd);
                float sqrtOneMinusAlpha = (float)Math.Sqrt(1.0f - alphaProd);

                sqrtAlphaProd[i] = sqrtAlpha;
                sqrtOneMinusAlphaProd[i] = sqrtOneMinusAlpha;
            }

            // Reshape sqrtAlphaProd and sqrtOneMinusAlphaProd to match the shape of originalSamples
            int[] outputShape = originalSamples.Dimensions.ToArray();
            outputShape[0] = timesteps.Length; // Update the batch size dimension
            sqrtAlphaProd = sqrtAlphaProd.Reshape(outputShape).ToDenseTensor();
            sqrtOneMinusAlphaProd = sqrtOneMinusAlphaProd.Reshape(outputShape).ToDenseTensor();

            // Compute noisy samples
            var noisySamples = new DenseTensor<float>(originalSamples.Dimensions);
            for (int i = 0; i < originalSamples.Length; i++)
            {
                noisySamples.SetValue(i, sqrtAlphaProd.GetValue(i) * originalSamples.GetValue(i) + sqrtOneMinusAlphaProd.GetValue(i) * noise.GetValue(i));
            }
            return noisySamples;
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
            float alphaProdT = _alphasCumulativeProducts[timestep];
            float alphaProdTPrev = prevTimestep >= 0 ? _alphasCumulativeProducts[prevTimestep] : 1.0f;
            float currentBetaT = 1 - alphaProdT / alphaProdTPrev;

            // For t > 0, compute predicted variance βt
            float variance = (1 - alphaProdTPrev) / (1 - alphaProdT) * currentBetaT;

            // Clamp variance to ensure it's not 0
            variance = Math.Max(variance, 1e-20f);


            if (SchedulerOptions.VarianceType ==  VarianceType.FixedSmallLog)
            {
                variance = (float)Math.Exp(0.5 * Math.Log(variance));
            }
            else if (SchedulerOptions.VarianceType == VarianceType.FixedLarge)
            {
                variance = currentBetaT;
            }
            else if (SchedulerOptions.VarianceType == VarianceType.FixedLargeLog)
            {
                variance = (float)Math.Log(currentBetaT);
            }
            else if (SchedulerOptions.VarianceType ==  VarianceType.Learned)
            {
                return predictedVariance;
            }
            else if (SchedulerOptions.VarianceType == VarianceType.LearnedRange)
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

            // Ensure the data type is float32 or float64
            if (sample.dtype != typeof(float) && sample.dtype != typeof(double))
            {
                sample = sample.astype(NPTypeCode.Single); // Upcast for quantile calculation
            }

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
            var index = (int)Math.Ceiling((percentile / 100f) * (data.Shape[0] - 1));

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
            return SchedulerOptions.VarianceType == VarianceType.Learned || SchedulerOptions.VarianceType == VarianceType.LearnedRange;
        }
    }
}

using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion3
{
    public class ControlNetDiffuser : StableDiffusion3Diffuser
    {
        protected readonly ControlNetModel _controlNet;

        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetDiffuser"/> class.
        /// </summary>
        /// <param name="controlNet">The control net.</param>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public ControlNetDiffuser(ControlNetModel controlNet, UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, logger)
        {
            _controlNet = controlNet;
        }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ControlNet;


        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="options"></param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var generateOptions = options.GenerateOptions;
            var schedulerOptions = generateOptions.SchedulerOptions;
            var performGuidance = ShouldPerformGuidance(schedulerOptions);
            var promptEmbedsCond = options.PromptEmbeddings.PromptEmbeds;
            var pooledPromptEmbedsCond = options.PromptEmbeddings.PooledPromptEmbeds;
            var promptEmbedsUncond = options.PromptEmbeddings.NegativePromptEmbeds;
            var pooledPromptEmbedsUncond = options.PromptEmbeddings.NegativePooledPromptEmbeds;

            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get Model metadata
                var metadata = await _unet.LoadAsync(cancellationToken: cancellationToken);

                // Get Model metadata
                var controlNetMetadata = await _controlNet.LoadAsync(cancellationToken: cancellationToken);

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Control Image
                var controlImage = await PrepareControlImage(generateOptions, cancellationToken);

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create inputs.
                    var inputTensor = scheduler.ScaleInput(latents, timestep);
                    var timestepTensor = CreateTimestepTensor(timestep);
                    var conditioningScaleTensor = CreateConditioningScaleTensor<double>(schedulerOptions.ConditioningScale);
                    var transformerOutputBuffer = schedulerOptions.GetScaledDimension(1, 16);
                    var controlNetOutputBuffer = new int[] { _controlNet.LayerCount, 4096, 1536 };

                    // Transformer Inference
                    var conditionalResult = await RunTransformerAsync(metadata, controlNetMetadata, inputTensor, timestepTensor, promptEmbedsCond, pooledPromptEmbedsCond, controlImage, conditioningScaleTensor, transformerOutputBuffer, controlNetOutputBuffer, cancellationToken);

                    // Classifier free guidance
                    if (performGuidance)
                    {
                        // Transformer Inference
                        var unconditionalResult = await RunTransformerAsync(metadata, controlNetMetadata, inputTensor, timestepTensor, promptEmbedsUncond, pooledPromptEmbedsUncond, controlImage, conditioningScaleTensor, transformerOutputBuffer, controlNetOutputBuffer, cancellationToken);
                        conditionalResult = PerformGuidance(conditionalResult, unconditionalResult, schedulerOptions.GuidanceScale);
                    }

                    // Scheduler Step
                    latents = scheduler.Step(conditionalResult, timestep, latents).Result;

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (generateOptions.IsLowMemoryComputeEnabled)
                    await Task.WhenAll(_controlNet.UnloadAsync(), _unet.UnloadAsync());

                // Decode Latents
                return await DecodeLatentsAsync(generateOptions, latents, cancellationToken);
            }
        }


        /// <summary>
        /// Run transformer inference
        /// </summary>
        /// <param name="transformerMetadata">The transformer metadata.</param>
        /// <param name="controlNetMetadata">The control net metadata.</param>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="timestepTensor">The timestep tensor.</param>
        /// <param name="promptEmbeds">The prompt embeds.</param>
        /// <param name="promptPooledEmbeds">The prompt pooled embeds.</param>
        /// <param name="controlTensor">The control tensor.</param>
        /// <param name="controlScaleTensor">The control scale tensor.</param>
        /// <param name="transformerOutputBuffer">The transformer output buffer.</param>
        /// <param name="controlNetOutputBuffer">The control net output buffer.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        protected async Task<DenseTensor<float>> RunTransformerAsync(
            OnnxMetadata transformerMetadata,
            OnnxMetadata controlNetMetadata,
            DenseTensor<float> inputTensor,
            DenseTensor<float> timestepTensor,
            DenseTensor<float> promptEmbeds,
            DenseTensor<float> promptPooledEmbeds,
            DenseTensor<float> controlTensor,
            DenseTensor<double> controlScaleTensor,
            int[] transformerOutputBuffer,
            int[] controlNetOutputBuffer,
            CancellationToken cancellationToken)
        {

            var controlNetPooledPromptEmbedsUncond = promptPooledEmbeds;
            if (_controlNet.DisablePooledProjection)
            {
                // Instantx SD3 ControlNet used zero pooled projection
                controlNetPooledPromptEmbedsUncond = new DenseTensor<float>(promptPooledEmbeds.Dimensions);
            }

            using (var transformerParams = new OnnxInferenceParameters(transformerMetadata, cancellationToken))
            using (var controlNetParams = new OnnxInferenceParameters(controlNetMetadata, cancellationToken))
            {
                // Transformer Inputs
                transformerParams.AddInputTensor(inputTensor);
                transformerParams.AddInputTensor(timestepTensor);
                transformerParams.AddInputTensor(promptEmbeds);
                transformerParams.AddInputTensor(promptPooledEmbeds);
                transformerParams.AddOutputBuffer(transformerOutputBuffer);

                // ControlNet Inputs
                controlNetParams.AddInputTensor(inputTensor);
                controlNetParams.AddInputTensor(timestepTensor);
                controlNetParams.AddInputTensor(promptEmbeds);
                controlNetParams.AddInputTensor(controlNetPooledPromptEmbedsUncond);
                controlNetParams.AddInputTensor(controlTensor);
                controlNetParams.AddInputTensor(controlScaleTensor);
                controlNetParams.AddOutputBuffer(controlNetOutputBuffer);

                // ControlNet Inference
                var controlNetResults = await _controlNet.RunInferenceAsync(controlNetParams);
                using (var controlNetResult = controlNetResults.First())
                {
                    transformerParams.AddInput(controlNetResult);

                    // Transformer Inference
                    var transformerResults = await _unet.RunInferenceAsync(transformerParams);
                    using (var transformerResult = transformerResults.FirstOrDefault())
                    {
                        return transformerResult.ToDenseTensor();
                    }
                }
            }
        }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
        {
            if (!options.Timesteps.IsNullOrEmpty())
                return options.Timesteps;

            return scheduler.Timesteps;
        }


        /// <summary>
        /// Prepares the input latents.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected override Task<DenseTensor<float>> PrepareLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            return Task.FromResult(scheduler.CreateRandomSample(options.SchedulerOptions.GetScaledDimension(1, 16)));
        }


        /// <summary>
        /// Creates the Conditioning Scale tensor.
        /// </summary>
        /// <param name="conditioningScale">The conditioningScale.</param>
        /// <returns></returns>
        protected static DenseTensor<T> CreateConditioningScaleTensor<T>(T conditioningScale) where T : unmanaged
        {
            return new DenseTensor<T>(new T[] { conditioningScale }, new int[] { 1 });
        }


        /// <summary>
        /// Prepares the control image.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns></returns>
        protected virtual async Task<DenseTensor<float>> PrepareControlImage(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            var controlImageTensor = await options.InputContolImage.GetImageTensorAsync(options.SchedulerOptions.Height, options.SchedulerOptions.Width);
            if (_controlNet.InvertInput)
                InvertInputTensor(controlImageTensor);

            var outputDimension = new[] { 1, 16, controlImageTensor.Dimensions[2] / 8, controlImageTensor.Dimensions[3] / 8 };
            var metadata = await _vaeEncoder.LoadAsync(cancellationToken: cancellationToken);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(controlImageTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);
                var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    var scaledSample = result.ToDenseTensor();

                    // Instantx SD3 ControlNet does not apply shift factor
                    if (!_controlNet.DisablePooledProjection)
                        scaledSample.Subtract(ShiftFactor);

                    return scaledSample.MultiplyBy(_vaeEncoder.ScaleFactor);
                }
            }
        }


        /// <summary>
        /// Inverts the input tensor.
        /// </summary>
        /// <param name="values">The values.</param>
        protected static void InvertInputTensor(DenseTensor<float> values)
        {
            for (int j = 0; j < values.Length; j++)
            {
                values.SetValue(j, 1f - values.GetValue(j));
            }
        }
    }
}

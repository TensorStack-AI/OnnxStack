using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Schedulers.StableDiffusion;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion3
{
    public abstract class StableDiffusion3Diffuser : StableDiffusionXLDiffuser
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion3Diffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        protected StableDiffusion3Diffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, logger) { }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableDiffusion3;

        /// <summary>
        /// Gets the shift factor.
        /// </summary>
        protected override float ShiftFactor => 0.0609f;


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
            var promptPooledEmbedsCond = options.PromptEmbeddings.PooledPromptEmbeds;
            var promptEmbedsUncond = options.PromptEmbeddings.NegativePromptEmbeds;
            var promptPooledEmbedsUncond = options.PromptEmbeddings.NegativePooledPromptEmbeds;

            var optimizations = GetOptimizations(generateOptions, options.PromptEmbeddings, progressCallback);
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get Model metadata
                var metadata = await _unet.LoadAsync(optimizations, cancellationToken);

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputTensor = scheduler.ScaleInput(latents, timestep);
                    var timestepTensor = CreateTimestepTensor(timestep);
                    var transformerOutputBuffer = schedulerOptions.GetScaledDimension(channels: 16);

                    // Transformer Inference
                    var conditionalResult = await RunTransformerAsync(metadata, inputTensor, timestepTensor, promptEmbedsCond, promptPooledEmbedsCond, transformerOutputBuffer, cancellationToken);

                    // Classifier free guidance
                    if (performGuidance)
                    {
                        // Transformer Inference
                        var unconditionalResult = await RunTransformerAsync(metadata, inputTensor, timestepTensor, promptEmbedsUncond, promptPooledEmbedsUncond, transformerOutputBuffer, cancellationToken);
                        conditionalResult = PerformGuidance(conditionalResult, unconditionalResult, schedulerOptions.GuidanceScale);
                    }

                    // Scheduler Step
                    latents = scheduler.Step(conditionalResult, timestep, latents).Result;

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (generateOptions.IsLowMemoryComputeEnabled)
                    await _unet.UnloadAsync();

                // Decode Latents
                return await DecodeLatentsAsync(generateOptions, latents, cancellationToken);
            }
        }


        /// <summary>
        /// Run transformer model inference
        /// </summary>
        /// <param name="transformerMetadata">The transformer metadata.</param>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="timestepTensor">The timestep tensor.</param>
        /// <param name="promptEmbeds">The prompt embeds.</param>
        /// <param name="promptPooledEmbeds">The prompt pooled embeds.</param>
        /// <param name="transformerOutputBuffer">The transformer output buffer.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        protected async Task<DenseTensor<float>> RunTransformerAsync(OnnxMetadata transformerMetadata, DenseTensor<float> inputTensor, DenseTensor<float> timestepTensor, DenseTensor<float> promptEmbeds, DenseTensor<float> promptPooledEmbeds, int[] transformerOutputBuffer, CancellationToken cancellationToken)
        {
            using (var transformerParams = new OnnxInferenceParameters(transformerMetadata, cancellationToken))
            {
                // Timestep could be located at index 1 or index 3
                var timestepIndex = transformerMetadata.Inputs.IndexOf(x => x.Dimensions.Length == 1);
                transformerParams.AddInputTensor(inputTensor);
                if (timestepIndex == 1)
                    transformerParams.AddInputTensor(timestepTensor);
                transformerParams.AddInputTensor(promptEmbeds);
                transformerParams.AddInputTensor(promptPooledEmbeds);
                if (timestepIndex > 1)
                    transformerParams.AddInputTensor(timestepTensor);
                transformerParams.AddOutputBuffer(transformerOutputBuffer);

                // Transformer Inference
                var transformerResults = await _unet.RunInferenceAsync(transformerParams);
                using (var transformerResult = transformerResults.FirstOrDefault())
                {
                    return transformerResult.ToDenseTensor();
                }
            }
        }


        /// <summary>
        /// Gets the scheduler.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="schedulerConfig">The scheduler configuration.</param>
        /// <returns></returns>
        protected override IScheduler GetScheduler(SchedulerOptions options)
        {
            return options.SchedulerType switch
            {
                SchedulerType.FlowMatchEulerDiscrete => new FlowMatchEulerDiscreteScheduler(options),
                SchedulerType.FlowMatchEulerDynamic => new FlowMatchEulerDynamicScheduler(options),
                _ => default
            };
        }


        /// <summary>
        /// Gets the optimizations.
        /// </summary>
        /// <param name="generateOptions">The generate options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <returns>OnnxOptimizations.</returns>
        protected virtual OnnxOptimizations GetOptimizations(GenerateOptions generateOptions, PromptEmbeddingsResult promptEmbeddings, IProgress<DiffusionProgress> progressCallback = null)
        {
            var optimizationLevel = generateOptions.OptimizationType == OptimizationType.None
                ? GraphOptimizationLevel.ORT_DISABLE_ALL
                : GraphOptimizationLevel.ORT_ENABLE_ALL;
            var optimizations = new OnnxOptimizations(optimizationLevel);

            // These are not added to the model graph but ensure model reloads on Width/Height
            optimizations.Add("dummy_width", generateOptions.SchedulerOptions.GetScaledWidth());
            optimizations.Add("dummy_height", generateOptions.SchedulerOptions.GetScaledHeight());
            if (generateOptions.OptimizationType >= OptimizationType.Level2)
            {
                optimizations.Add("transformer_sample_batch", 1);
                optimizations.Add("transformer_sample_channels", 16);
                optimizations.Add("transformer_time_batch", 1);
                optimizations.Add("transformer_hidden_batch", 1);
                optimizations.Add("transformer_text_embeds_batch", 1);
                optimizations.Add("transformer_text_embeds_size", promptEmbeddings.PooledPromptEmbeds.Dimensions[1]);
            }
            if (generateOptions.OptimizationType >= OptimizationType.Level3)
            {
                optimizations.Add("transformer_sample_width", generateOptions.SchedulerOptions.GetScaledWidth());
                optimizations.Add("transformer_sample_height", generateOptions.SchedulerOptions.GetScaledHeight());
            }
            if (generateOptions.OptimizationType >= OptimizationType.Level4)
            {
                optimizations.Add("transformer_hidden_size", promptEmbeddings.PromptEmbeds.Dimensions[1]);
            }

            if (_unet.HasOptimizationsChanged(optimizations))
            {
                progressCallback.Notify("Optimizing Pipeline...");
            }
            return optimizations;
        }

    }
}

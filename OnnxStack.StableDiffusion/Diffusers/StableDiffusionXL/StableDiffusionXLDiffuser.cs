using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Models;
using OnnxStack.StableDiffusion.Schedulers.LatentConsistency;
using OnnxStack.StableDiffusion.Schedulers.StableDiffusion;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL
{
    public abstract class StableDiffusionXLDiffuser : DiffuserBase
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionXLDiffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusionXLDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, logger) { }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableDiffusionXL;

        /// <summary>
        /// Gets the shift factor.
        /// </summary>
        protected virtual float ShiftFactor => 0;

        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="options">The options.</param>
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

                // Get Time ids
                var addTimeIds = GetAddTimeIds(schedulerOptions);

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

                    var outputDimensions = schedulerOptions.GetScaledDimension();
                    using (var unetCondParams = new OnnxInferenceParameters(metadata, cancellationToken))
                    using (var unetUncondParams = new OnnxInferenceParameters(metadata, cancellationToken))
                    {
                        unetCondParams.AddInputTensor(inputTensor);
                        unetCondParams.AddInputTensor(timestepTensor);
                        unetCondParams.AddInputTensor(promptEmbedsCond);
                        unetCondParams.AddInputTensor(pooledPromptEmbedsCond);
                        unetCondParams.AddInputTensor(addTimeIds);
                        unetCondParams.AddOutputBuffer(outputDimensions);

                        var unetUncondResults = default(IReadOnlyCollection<OrtValue>);
                        if (performGuidance)
                        {
                            unetUncondParams.AddInputTensor(inputTensor);
                            unetUncondParams.AddInputTensor(timestepTensor);
                            unetUncondParams.AddInputTensor(promptEmbedsUncond);
                            unetUncondParams.AddInputTensor(pooledPromptEmbedsUncond);
                            unetUncondParams.AddInputTensor(addTimeIds);
                            unetUncondParams.AddOutputBuffer(outputDimensions);
                            unetUncondResults = await _unet.RunInferenceAsync(unetUncondParams);
                        }

                        var unetCondResults = await _unet.RunInferenceAsync(unetCondParams);
                        using (var unetCondResult = unetCondResults.First())
                        using (var unetUncondResult = unetUncondResults?.FirstOrDefault())
                        {
                            var noisePred = unetCondResult.ToDenseTensor();

                            // Perform guidance
                            if (performGuidance)
                                noisePred = PerformGuidance(noisePred, unetUncondResult.ToDenseTensor(), schedulerOptions.GuidanceScale);

                            // Scheduler Step
                            latents = scheduler.Step(noisePred, timestep, latents).Result;
                        }
                    }

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
        /// Performs the classifier-free guidance.
        /// </summary>
        /// <param name="noisePredCond">The noise pred cond.</param>
        /// <param name="noisePredUncond">The noise pred uncond.</param>
        /// <param name="guidanceScale">The guidance scale.</param>
        /// <returns></returns>
        protected DenseTensor<float> PerformGuidance(DenseTensor<float> noisePredCond, DenseTensor<float> noisePredUncond, float guidanceScale)
        {
            // Perform guidance
            noisePredUncond.Lerp(noisePredCond, guidanceScale);
            return noisePredUncond;
        }


        /// <summary>
        /// Gets the add AddTimeIds.
        /// </summary>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        protected DenseTensor<float> GetAddTimeIds(SchedulerOptions schedulerOptions)
        {
            float[] result = _unet.ModelType == ModelType.Refiner
                ? [schedulerOptions.Height, schedulerOptions.Width, 0, 0, schedulerOptions.AestheticScore]
                : [schedulerOptions.Height, schedulerOptions.Width, 0, 0, schedulerOptions.Height, schedulerOptions.Width];
            return new DenseTensor<float>(result, [1, result.Length]);
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
                SchedulerType.LMS => new LMSScheduler(options),
                SchedulerType.Euler => new EulerScheduler(options),
                SchedulerType.EulerAncestral => new EulerAncestralScheduler(options),
                SchedulerType.DDPM => new DDPMScheduler(options),
                SchedulerType.DDIM => new DDIMScheduler(options),
                SchedulerType.KDPM2 => new KDPM2Scheduler(options),
                SchedulerType.KDPM2Ancestral => new KDPM2AncestralScheduler(options),
                SchedulerType.LCM => new LCMScheduler(options),
                _ => default
            };
        }


        /// <summary>
        /// Gets the optimizations.
        /// </summary>
        /// <param name="generateOptions">The generate options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <returns>OnnxOptimizations.</returns>
        private OnnxOptimizations GetOptimizations(GenerateOptions generateOptions, PromptEmbeddingsResult promptEmbeddings, IProgress<DiffusionProgress> progressCallback = null)
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
                optimizations.Add("unet_sample_batch", 1);
                optimizations.Add("unet_sample_channels", 4);
                optimizations.Add("unet_time_batch", 1);
                optimizations.Add("unet_hidden_batch", 1);
                optimizations.Add("unet_text_embeds_batch", 1);
                optimizations.Add("unet_time_ids", 1);
                optimizations.Add("unet_time_ids_size", 6);
                optimizations.Add("unet_text_embeds_size", 1280);
            }
            if (generateOptions.OptimizationType >= OptimizationType.Level3)
            {
                optimizations.Add("unet_sample_width", generateOptions.SchedulerOptions.GetScaledWidth());
                optimizations.Add("unet_sample_height", generateOptions.SchedulerOptions.GetScaledHeight());
            }
            if (generateOptions.OptimizationType >= OptimizationType.Level4)
            {
                optimizations.Add("unet_hidden_sequence", promptEmbeddings.PromptEmbeds.Dimensions[1]);
            }

            if (_unet.HasOptimizationsChanged(optimizations))
            {
                progressCallback.Notify("Optimizing Pipeline...");
            }

            return optimizations;
        }


        /// <summary>
        /// Decodes the latents.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        protected override async Task<DenseTensor<float>> DecodeLatentsAsync(GenerateOptions options, DenseTensor<float> latents, CancellationToken cancellationToken = default)
        {
            var timestamp = _logger.LogBegin();
            latents = latents
                .MultiplyBy(1.0f / _vaeDecoder.ScaleFactor)
                .Add(ShiftFactor);

            try
            {
                if (options.IsAutoEncoderTileEnabled)
                    return await DecodeLatentsTilesAsync(latents, options, cancellationToken);

                return await DecodeLatentsAsync(latents, options, cancellationToken);
            }
            finally
            {
                if (options.IsLowMemoryDecoderEnabled)
                    await _vaeDecoder.UnloadAsync();

                _logger?.LogEnd(LogLevel.Debug, "VaeDecoder", timestamp);
            }
        }


        /// <summary>
        /// Decode latents
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        protected virtual async Task<DenseTensor<float>> DecodeLatentsAsync(DenseTensor<float> latents, GenerateOptions options, CancellationToken cancellationToken = default)
        {
            var outputDim = new[] { 1, 3, latents.Dimensions[2] * 8, latents.Dimensions[3] * 8 };
            var metadata = await _vaeDecoder.LoadAsync(cancellationToken: cancellationToken);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(latents);
                inferenceParameters.AddOutputBuffer(outputDim);
                var results = await _vaeDecoder.RunInferenceAsync(inferenceParameters);
                using (var imageResult = results.First())
                {
                    return imageResult.ToDenseTensor();
                }
            }
        }


        /// <summary>
        /// Decode latents as tiles 
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;DenseTensor`1&gt; representing the asynchronous operation.</returns>
        private async Task<DenseTensor<float>> DecodeLatentsTilesAsync(DenseTensor<float> imageTensor, GenerateOptions options, CancellationToken cancellationToken = default)
        {
            var tileSize = 64;
            var scaleFactor = 8;
            var width = imageTensor.Dimensions[3];
            var height = imageTensor.Dimensions[2];
            var tileMode = options.AutoEncoderTileMode;
            var tileOverlap = options.AutoEncoderTileOverlap;
            if (width <= (tileSize + tileOverlap) || height <= (tileSize + tileOverlap))
                return await DecodeLatentsAsync(imageTensor, options, cancellationToken);

            var inputTiles = new ImageTiles(imageTensor, tileMode, tileOverlap);
            var outputTiles = new ImageTiles
            (
                inputTiles.Width * scaleFactor,
                inputTiles.Height * scaleFactor,
                tileMode,
                inputTiles.Overlap * scaleFactor,
                await DecodeLatentsTilesAsync(inputTiles.Tile1, options, cancellationToken),
                await DecodeLatentsTilesAsync(inputTiles.Tile2, options, cancellationToken),
                await DecodeLatentsTilesAsync(inputTiles.Tile3, options, cancellationToken),
                await DecodeLatentsTilesAsync(inputTiles.Tile4, options, cancellationToken)
            );
            return outputTiles.JoinTiles();
        }

    }
}

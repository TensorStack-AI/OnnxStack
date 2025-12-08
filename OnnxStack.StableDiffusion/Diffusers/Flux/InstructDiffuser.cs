using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
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

namespace OnnxStack.StableDiffusion.Diffusers.Flux
{
    public sealed class InstructDiffuser : FluxDiffuser
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="InstructDiffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public InstructDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, logger) { }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ImageToImage;


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
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public override async Task<DenseTensor<float>> DiffuseAsync(DiffuseOptions options, IProgress<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var generateOptions = options.GenerateOptions;
            var schedulerOptions = generateOptions.SchedulerOptions;
            var performGuidance = ShouldPerformGuidance(schedulerOptions);
            var conditionalEmbeds = options.PromptEmbeddings.PromptEmbeds;
            var conditionalPooledEmbeds = options.PromptEmbeddings.PooledPromptEmbeds;
            var unconditionalEmbeds = options.PromptEmbeddings.NegativePromptEmbeds;
            var unconditionalPooledEmbeds = options.PromptEmbeddings.NegativePooledPromptEmbeds;

            var optimizations = GetOptimizations(generateOptions, options.PromptEmbeddings, progressCallback);
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                progressCallback.Notify("Prepare Input...");

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latents = PackLatents(scheduler.CreateRandomSample(schedulerOptions.GetScaledDimension(channels: 16)));

                // Create image latents
                var imageLatents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Create ImageIds
                var imgIds = PrepareImageLatents(generateOptions);

                // Create TextIds
                var txtIds = PrepareLatentTextIds(conditionalEmbeds);

                // Create LatentIds
                var latentIds = PrepareLatentImageIds(schedulerOptions).Concatenate(imgIds);

                // Get Model metadata
                var metadata = await _unet.LoadAsync(optimizations, cancellationToken);

                // Guiadance
                var guidanceTensor = new DenseTensor<float>(new float[] { schedulerOptions.GuidanceScale2 }, [1]);

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputTensor = latents.Concatenate(imageLatents, 1);
                    var timestepTensor = CreateTimestepTensor(timestep);

                    // Transformer Inference
                    var conditionalResult = await RunTransformerAsync
                    (
                        metadata,
                        inputTensor,
                        timestepTensor,
                        conditionalEmbeds,
                        conditionalPooledEmbeds,
                        guidanceTensor,
                        latentIds,
                        txtIds,
                        cancellationToken
                    );
                    conditionalResult = RemoveImageLatents(conditionalResult, latents);

                    // Scheduler Step
                    latents = scheduler.Step(conditionalResult, timestep, latents).Result;

                    ReportProgress(progressCallback, "Step", step, timesteps.Count, stepTime, UnpackLatents(latents, schedulerOptions.Width, schedulerOptions.Height));
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (generateOptions.IsLowMemoryComputeEnabled)
                    await _unet.UnloadAsync();

                // Decode Latents
                return await DecodeLatentsAsync(generateOptions, UnpackLatents(latents, schedulerOptions.Width, schedulerOptions.Height), cancellationToken);
            }
        }


        /// <summary>
        /// Prepares the latents for inference.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override async Task<DenseTensor<float>> PrepareLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            var imageWidth = Math.Min(options.InputImage.Width, options.SchedulerOptions.Width);
            var imageHeight = Math.Min(options.InputImage.Height, options.SchedulerOptions.Height);
            var imageTensor = await options.InputImage.GetImageTensorAsync(imageHeight, imageWidth);
            var metadata = await _vaeEncoder.LoadAsync(cancellationToken: cancellationToken);
            int[] outputDimension = [1, VaeChannels, imageHeight * 2 / VaeChannels, imageWidth * 2 / VaeChannels];
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);

                var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    // Unload if required
                    if (options.IsLowMemoryEncoderEnabled)
                        await _vaeEncoder.UnloadAsync();

                    var scaledSample = result
                        .ToDenseTensor()
                        .SubtractFloat(ShiftFactor)
                        .MultiplyBy(_vaeEncoder.ScaleFactor);
                    return PackLatents(scaledSample);
                }
            }
        }


        /// <summary>
        /// Prepare the image latent ids
        /// </summary>
        /// <param name="options">The GenerateOptions</param>
        private DenseTensor<float> PrepareImageLatents(GenerateOptions options)
        {
            var imageWidth = Math.Min(options.InputImage.Width, options.SchedulerOptions.Width);
            var imageHeight = Math.Min(options.InputImage.Height, options.SchedulerOptions.Height);
            var latentImageIds = PrepareLatentImageIds(new SchedulerOptions { Height = imageHeight, Width = imageWidth });

            // first dimension set to 1 instead of 0
            for (int i = 0; i < latentImageIds.Dimensions[0]; i++)
                latentImageIds[i, 0] = 1;

            return latentImageIds;
        }


        /// <summary>
        /// Removes the image latents from the prediction result.
        /// </summary>
        /// <param name="prediction">The prediction.</param>
        /// <param name="latents">The latents.</param>
        /// <returns>DenseTensor&lt;System.Single&gt;.</returns>
        private DenseTensor<float> RemoveImageLatents(DenseTensor<float> prediction, DenseTensor<float> latents)
        {
            var latentSize = latents.Dimensions[1];
            var sliceSize = prediction.Dimensions[2];
            var result = new DenseTensor<float>(new[] { 1, latentSize, sliceSize });
            var totalElementsToCopy = latentSize * sliceSize;
            prediction.Buffer.Span[..totalElementsToCopy].CopyTo(result.Buffer.Span);
            return result;
        }


        /// <summary>
        /// Gets the optimizations.
        /// </summary>
        /// <param name="generateOptions">The generate options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="progressCallback">The progress callback.</param>
        private OnnxOptimizations GetOptimizations(GenerateOptions generateOptions, PromptEmbeddingsResult promptEmbeddings, IProgress<DiffusionProgress> progressCallback = null)
        {
            var optimizationLevel = generateOptions.OptimizationType == OptimizationType.None
                ? GraphOptimizationLevel.ORT_DISABLE_ALL
                : GraphOptimizationLevel.ORT_ENABLE_ALL;
            var optimizations = new OnnxOptimizations(optimizationLevel);
            // These are not added to the model graph but ensure model reloads on Width/Height
            optimizations.Add("dummy_width", generateOptions.SchedulerOptions.GetScaledWidth());
            optimizations.Add("dummy_height", generateOptions.SchedulerOptions.GetScaledHeight());
            if (_unet.HasOptimizationsChanged(optimizations))
            {
                progressCallback.Notify("Optimizing Pipeline...");
            }

            return optimizations;
        }

    }
}

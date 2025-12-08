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

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion
{
    public class InstructDiffuser : StableDiffusionDiffuser
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
        /// Prepares the latents for inference.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected override Task<DenseTensor<float>> PrepareLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            return Task.FromResult(scheduler.CreateRandomSample(options.SchedulerOptions.GetScaledDimension(), scheduler.InitNoiseSigma));
        }


        /// <summary>
        /// Prepares the image latents for inference.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected async Task<DenseTensor<float>> PrepareImageLatentsAsync(GenerateOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps, CancellationToken cancellationToken = default)
        {
            var imageTensor = await options.InputImage.GetImageTensorAsync(options.SchedulerOptions.Height, options.SchedulerOptions.Width);
            var outputDimension = options.SchedulerOptions.GetScaledDimension();
            var metadata = await _vaeEncoder.LoadAsync(cancellationToken: cancellationToken);
            using (var inferenceParameters = new OnnxInferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(outputDimension);

                var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    if (options.IsLowMemoryEncoderEnabled)
                        await _vaeEncoder.UnloadAsync();

                    var imageLatents = result.ToDenseTensor();
                    return imageLatents
                        .Repeat(2)
                        .Concatenate(new DenseTensor<float>(imageLatents.Dimensions));
                }
            }
        }


        /// <summary>
        /// Prepares the prompt embeds.
        /// </summary>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <returns></returns>
        protected DenseTensor<float> PreparePromptEmbeds(DenseTensor<float> promptEmbeds)
        {
            var embeds = promptEmbeds
                .SplitBatch()
                .ToArray();
            var uncondEmbeds = embeds[0];
            var condEmbeds = embeds[1];
            return new[] { condEmbeds, uncondEmbeds, uncondEmbeds }.Join();
        }


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
            var promptEmbeds = options.PromptEmbeddings.GetPromptEmbeds(performGuidance);
            var pooledPromptEmbeds = options.PromptEmbeddings.GetPooledPromptEmbeds(performGuidance);
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get Unet Model metadata
                var unetMetadata = await _unet.LoadAsync(cancellationToken: cancellationToken);

                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create Noise Latents
                progressCallback.Notify("Prepare Input...");
                var latents = await PrepareLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Create Image Latents
                var imageLatents = await PrepareImageLatentsAsync(generateOptions, scheduler, timesteps, cancellationToken);

                // Loop though the timesteps
                var step = 0;
                ReportProgress(progressCallback, "Step", 0, timesteps.Count, 0);
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Prepare latents
                    var batchCount = performGuidance ? 3 : 1;
                    var inputLatents = performGuidance ? latents.Repeat(batchCount) : latents;
                    var scaledLatents = scheduler.ScaleInput(inputLatents, timestep);

                    // Create input tensors.
                    var inputTensor = scaledLatents.Concatenate(imageLatents, 1);
                    var promptTensor = PreparePromptEmbeds(promptEmbeds);
                    var timestepTensor = CreateTimestepTensor(timestep);

                    var outputDimension = schedulerOptions.GetScaledDimension(batchCount);
                    using (var inferenceParameters = new OnnxInferenceParameters(unetMetadata, cancellationToken))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptTensor);
                        inferenceParameters.AddOutputBuffer(outputDimension);

                        // Unet Inference
                        var results = await _unet.RunInferenceAsync(inferenceParameters);
                        using (var result = results.First())
                        {
                            var noisePred = result.ToDenseTensor();

                            // Perform guidance
                            if (performGuidance)
                                noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale, schedulerOptions.Strength);

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
        /// Performs the guidance.
        /// </summary>
        /// <param name="noisePrediction">The noise prediction.</param>
        /// <param name="guidanceScale">The guidance scale.</param>
        /// <param name="imageGuidanceScale">The image guidance scale.</param>
        /// <returns></returns>
        protected DenseTensor<float> PerformGuidance(DenseTensor<float> noisePrediction, float guidanceScale, float imagestrength)
        {
            // Split Prompt and Negative Prompt predictions
            var imageGuidance = (1f - imagestrength) + 1f;
            var predictions = noisePrediction.SplitBatch().ToArray();
            var noisePredText = predictions[0];
            var noisePredImage = predictions[1];
            var noisePredUncond = predictions[2];

            return noisePredUncond
                .Add(noisePredText
                    .Subtract(noisePredImage)
                    .MultiplyBy(guidanceScale)
                    .Add(noisePredImage
                        .Subtract(noisePredUncond)
                        .MultiplyBy(imageGuidance)));
        }

    }
}

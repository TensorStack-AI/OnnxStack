using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Image;
using OnnxStack.Core.Model;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusionXL
{
    public sealed class InpaintLegacyDiffuser : StableDiffusionXLDiffuser
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="InpaintLegacyDiffuser"/> class.
        /// </summary>
        /// <param name="unet"></param>
        /// <param name="vaeDecoder"></param>
        /// <param name="vaeEncoder"></param>
        /// <param name="logger"></param>
        public InpaintLegacyDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, MemoryModeType memoryMode, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, memoryMode, logger) { }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ImageInpaintLegacy;


        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="modelOptions">The model options.</param>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latentsOriginal = await PrepareLatentsAsync(promptOptions, schedulerOptions, scheduler, timesteps);

                // Create masks sample
                var maskImage = PrepareMask(promptOptions, schedulerOptions);

                // Generate some noise
                var noise = scheduler.CreateRandomSample(latentsOriginal.Dimensions);

                // Add noise to original latent
                var latents = scheduler.AddNoise(latentsOriginal, noise, timesteps);

                // Get Model metadata
                var metadata = await _unet.GetMetadataAsync();

                // Get Time ids
                var addTimeIds = GetAddTimeIds(schedulerOptions);

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputLatent = performGuidance ? latents.Repeat(2) : latents;
                    var inputTensor = scheduler.ScaleInput(inputLatent, timestep);
                    var timestepTensor = CreateTimestepTensor(timestep);
                    var timeids = performGuidance ? addTimeIds.Repeat(2) : addTimeIds;

                    var outputChannels = performGuidance ? 2 : 1;
                    var outputDimension = schedulerOptions.GetScaledDimension(outputChannels);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PooledPromptEmbeds);
                        inferenceParameters.AddInputTensor(timeids);
                        inferenceParameters.AddOutputBuffer(outputDimension);

                        var results = await _unet.RunInferenceAsync(inferenceParameters);
                        using (var result = results.First())
                        {
                            var noisePred = result.ToDenseTensor();

                            // Perform guidance
                            if (performGuidance)
                                noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale);

                            // Scheduler Step
                            var steplatents = scheduler.Step(noisePred, timestep, latents).Result;

                            // Add noise to original latent
                            var initLatentsProper = scheduler.AddNoise(latentsOriginal, noise, new[] { timestep });

                            // Apply mask and combine 
                            latents = ApplyMaskedLatents(steplatents, initLatentsProper, maskImage);
                        }
                    }

                    ReportProgress(progressCallback, step, timesteps.Count, latents);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (_memoryMode == MemoryModeType.Minimum)
                    await _unet.UnloadAsync();

                // Decode Latents
                return await DecodeLatentsAsync(promptOptions, schedulerOptions, latents);
            }
        }


        /// <summary>
        /// Gets the timesteps.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override IReadOnlyList<int> GetTimesteps(SchedulerOptions options, IScheduler scheduler)
        {
            if (!options.Timesteps.IsNullOrEmpty())
                return options.Timesteps;

            var inittimestep = Math.Min((int)(options.InferenceSteps * options.Strength), options.InferenceSteps);
            var start = Math.Max(options.InferenceSteps - inittimestep, 0);
            return scheduler.Timesteps.Skip(start).ToList();
        }


        /// <summary>
        /// Prepares the latents for inference.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        protected override async Task<DenseTensor<float>> PrepareLatentsAsync(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            var imageTensor = await prompt.InputImage.GetImageTensorAsync(options.Height, options.Width);

            //TODO: Model Config, Channels
            var outputDimensions = options.GetScaledDimension();
            var metadata = await _vaeEncoder.GetMetadataAsync();
            using (var inferenceParameters = new OnnxInferenceParameters(metadata))
            {
                inferenceParameters.AddInputTensor(imageTensor);
                inferenceParameters.AddOutputBuffer(outputDimensions);

                var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                using (var result = results.First())
                {
                    // Unload if required
                    if (_memoryMode == MemoryModeType.Minimum)
                        await _vaeEncoder.UnloadAsync();

                    var outputResult = result.ToDenseTensor();
                    var scaledSample = outputResult.MultiplyBy(_vaeEncoder.ScaleFactor);
                    return scaledSample;
                }
            }
        }


        /// <summary>
        /// Prepares the mask.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        private DenseTensor<float> PrepareMask(PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            using (var mask = promptOptions.InputImageMask.GetImage())
            {
                // Prepare the mask
                int width = schedulerOptions.GetScaledWidth();
                int height = schedulerOptions.GetScaledHeight();
                mask.Mutate(x => x.Grayscale());
                mask.Mutate(x => x.Resize(new Size(width, height), KnownResamplers.NearestNeighbor, true));
                var maskTensor = new DenseTensor<float>(new[] { 1, 4, width, height });
                mask.ProcessPixelRows(img =>
                {
                    for (int x = 0; x < width; x++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            var pixelSpan = img.GetRowSpan(y);
                            var value = 1f - (pixelSpan[x].A / 255.0f);
                            maskTensor[0, 0, y, x] = value;
                            maskTensor[0, 1, y, x] = value; // Needed for shape only
                            maskTensor[0, 2, y, x] = value; // Needed for shape only
                            maskTensor[0, 3, y, x] = value; // Needed for shape only
                        }
                    }
                });
                return maskTensor;
            }
        }


        /// <summary>
        /// Applies the masked latents.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="initLatentsProper">The initialize latents proper.</param>
        /// <param name="mask">The mask.</param>
        /// <returns></returns>
        private DenseTensor<float> ApplyMaskedLatents(DenseTensor<float> latents, DenseTensor<float> initLatentsProper, DenseTensor<float> mask)
        {
            var result = new DenseTensor<float>(latents.Dimensions);
            for (int i = 0; i < result.Length; i++)
            {
                float maskValue = mask.GetValue(i);
                result.SetValue(i, initLatentsProper.GetValue(i) * maskValue + latents.GetValue(i) * (1f - maskValue));
            }
            return result;
        }
    }
}

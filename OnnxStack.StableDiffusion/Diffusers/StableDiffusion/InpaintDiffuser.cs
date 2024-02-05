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


namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion
{
    public sealed class InpaintDiffuser : StableDiffusionDiffuser
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="InpaintDiffuser"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="vaeDecoder">The vae decoder.</param>
        /// <param name="vaeEncoder">The vae encoder.</param>
        /// <param name="logger">The logger.</param>
        public InpaintDiffuser(UNetConditionModel unet, AutoEncoderModel vaeDecoder, AutoEncoderModel vaeEncoder, MemoryModeType memoryMode, ILogger logger = default)
            : base(unet, vaeDecoder, vaeEncoder, memoryMode, logger) { }


        /// <summary>
        /// Gets the type of the diffuser.
        /// </summary>
        public override DiffuserType DiffuserType => DiffuserType.ImageInpaint;


        /// <summary>
        /// Runs the scheduler steps.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <param name="promptEmbeddings">The prompt embeddings.</param>
        /// <param name="performGuidance">if set to <c>true</c> [perform guidance].</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(PromptOptions promptOptions, SchedulerOptions schedulerOptions, PromptEmbeddingsResult promptEmbeddings, bool performGuidance, Action<DiffusionProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            // Get Scheduler
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latents = await PrepareLatentsAsync(promptOptions, schedulerOptions, scheduler, timesteps);

                // Create Image Mask
                var maskImage = PrepareMask(promptOptions, schedulerOptions);

                // Create Masked Image Latents
                var maskedImage = await PrepareImageMask(promptOptions, schedulerOptions);

                // Get Model metadata
                var metadata = await _unet.GetMetadataAsync();

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
                    inputTensor = ConcatenateLatents(inputTensor, maskedImage, maskImage);
                    var timestepTensor = CreateTimestepTensor(timestep);

                    var outputChannels = performGuidance ? 2 : 1;
                    var outputDimension = schedulerOptions.GetScaledDimension(outputChannels);
                    using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                    {
                        inferenceParameters.AddInputTensor(inputTensor);
                        inferenceParameters.AddInputTensor(timestepTensor);
                        inferenceParameters.AddInputTensor(promptEmbeddings.PromptEmbeds);
                        inferenceParameters.AddOutputBuffer(outputDimension);

                        var results = await _unet.RunInferenceAsync(inferenceParameters);
                        using (var result = results.First())
                        {
                            var noisePred = result.ToDenseTensor();

                            // Perform guidance
                            if (performGuidance)
                                noisePred = PerformGuidance(noisePred, schedulerOptions.GuidanceScale);

                            // Scheduler Step
                            latents = scheduler.Step(noisePred, timestep, latents).Result;
                        }
                    }

                    ReportProgress(progressCallback, step, timesteps.Count, latents);
                    _logger?.LogEnd($"Step {step}/{timesteps.Count}", stepTime);
                }

                // Unload if required
                if (_memoryMode != MemoryModeType.Maximum)
                    await _unet.UnloadAsync();

                // Decode Latents
                return await DecodeLatentsAsync(promptOptions, schedulerOptions, latents);
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
            using (var imageMask = promptOptions.InputImageMask.ToImage())
            {
                var width = schedulerOptions.GetScaledWidth();
                var height = schedulerOptions.GetScaledHeight();
                var imageTensor = new DenseTensor<float>(new[] { 1, 1, width, height });

                // Prepare the image
                imageMask.Mutate(x => x.Resize(new Size(width, height)));
                imageMask.Mutate(x => x.Grayscale());
                imageMask.ProcessPixelRows(img =>
                {
                    for (int x = 0; x < width; x++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            var pixelSpan = img.GetRowSpan(y);
                            var value = pixelSpan[x].A / 255.0f;
                            if (value < 0.5f)
                                value = 0f;
                            else if (value >= 0.5f)
                                value = 1f;

                            imageTensor[0, 0, y, x] = value;
                        }
                    }
                });

                imageTensor = imageTensor.MultiplyBy(_vaeEncoder.ScaleFactor);
                if (schedulerOptions.GuidanceScale > 1f)
                    imageTensor = imageTensor.Repeat(2);

                return imageTensor;
            }
        }


        /// <summary>
        /// Prepares the image mask.
        /// </summary>
        /// <param name="promptOptions">The prompt options.</param>
        /// <param name="schedulerOptions">The scheduler options.</param>
        /// <returns></returns>
        private async Task<DenseTensor<float>> PrepareImageMask(PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            using (var image = await promptOptions.InputImage.ToImageAsync())
            using (var mask = await promptOptions.InputImageMask.ToImageAsync())
            {
                int width = schedulerOptions.Width;
                int height = schedulerOptions.Height;

                // Prepare the image
                var imageTensor = new DenseTensor<float>(new[] { 1, 3, width, height });
                image.Mutate(x => x.Resize(new Size(width, height)));
                image.ProcessPixelRows(img =>
                {
                    for (int x = 0; x < width; x++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            var pixelSpan = img.GetRowSpan(y);
                            imageTensor[0, 0, y, x] = pixelSpan[x].R / 127.5f - 1f;
                            imageTensor[0, 1, y, x] = pixelSpan[x].G / 127.5f - 1f;
                            imageTensor[0, 2, y, x] = pixelSpan[x].B / 127.5f - 1f;
                        }
                    }
                });


                // Prepare the mask
                var imageMaskedTensor = new DenseTensor<float>(new[] { 1, 3, width, height });
                mask.Mutate(x => x.Resize(new Size(width, height)));
                mask.Mutate(x => x.Grayscale());
                mask.ProcessPixelRows(img =>
                {
                    for (int x = 0; x < width; x++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            var pixelSpan = img.GetRowSpan(y);
                            var value = pixelSpan[x].PackedValue;
                            var mask = value >= 127.5f;
                            imageMaskedTensor[0, 0, y, x] = mask ? 0f : imageTensor[0, 0, y, x];
                            imageMaskedTensor[0, 1, y, x] = mask ? 0f : imageTensor[0, 1, y, x];
                            imageMaskedTensor[0, 2, y, x] = mask ? 0f : imageTensor[0, 2, y, x];
                        }
                    }
                });

                // Encode the image
                var outputDimension = schedulerOptions.GetScaledDimension();
                var metadata = await _vaeEncoder.GetMetadataAsync();
                using (var inferenceParameters = new OnnxInferenceParameters(metadata))
                {
                    inferenceParameters.AddInputTensor(imageTensor);
                    inferenceParameters.AddOutputBuffer(outputDimension);

                    var results = await _vaeEncoder.RunInferenceAsync(inferenceParameters);
                    using (var result = results.First())
                    {
                        // Unload if required
                        if (_memoryMode != MemoryModeType.Maximum)
                            await _vaeEncoder.UnloadAsync();

                        var sample = result.ToDenseTensor();
                        var scaledSample = sample.MultiplyBy(_vaeEncoder.ScaleFactor);
                        if (schedulerOptions.GuidanceScale > 1f)
                            scaledSample = scaledSample.Repeat(2);

                        return scaledSample;
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
            return scheduler.Timesteps;
        }


        /// <summary>
        /// Prepares the latents.
        /// </summary>
        /// <param name="prompt">The prompt.</param>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="timesteps">The timesteps.</param>
        /// <returns></returns>
        protected override Task<DenseTensor<float>> PrepareLatentsAsync(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            return Task.FromResult(scheduler.CreateRandomSample(options.GetScaledDimension(), scheduler.InitNoiseSigma));
        }


        /// <summary>
        /// Concatenates the latents.
        /// </summary>
        /// <param name="input1">The input1.</param>
        /// <param name="input2">The input2.</param>
        /// <param name="input3">The input3.</param>
        /// <returns></returns>
        private DenseTensor<float> ConcatenateLatents(DenseTensor<float> input1, DenseTensor<float> input2, DenseTensor<float> input3)
        {
            int batch = input1.Dimensions[0];
            int height = input1.Dimensions[2];
            int width = input1.Dimensions[3];
            int channels = input1.Dimensions[1] + input3.Dimensions[1] + input2.Dimensions[1];
            var concatenated = new DenseTensor<float>(new[] { batch, channels, height, width });
            for (int i = 0; i < batch; i++)
            {
                for (int j = 0; j < channels; j++)
                {
                    if (j < input1.Dimensions[1])
                    {
                        // Copy from input1
                        for (int k = 0; k < height; k++)
                        {
                            for (int l = 0; l < width; l++)
                            {
                                concatenated[i, j, k, l] = input1[i, j, k, l];
                            }
                        }
                    }
                    else if (j < input1.Dimensions[1] + input3.Dimensions[1])
                    {
                        // Copy from input2
                        for (int k = 0; k < height; k++)
                        {
                            for (int l = 0; l < width; l++)
                            {
                                concatenated[i, j, k, l] = input3[i, j - input1.Dimensions[1], k, l];
                            }
                        }
                    }
                    else
                    {
                        // Copy from input3
                        for (int k = 0; k < height; k++)
                        {
                            for (int l = 0; l < width; l++)
                            {
                                concatenated[i, j, k, l] = input2[i, j - input1.Dimensions[1] - input3.Dimensions[1], k, l];
                            }
                        }
                    }
                }
            }
            return concatenated;
        }

    }
}

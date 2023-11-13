using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
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
    public sealed class InpaintLegacyDiffuser : StableDiffusionDiffuser
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InpaintLegacyDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public InpaintLegacyDiffuser(IOnnxModelService onnxModelService, IPromptService promptService, ILogger<StableDiffusionDiffuser> logger)
            : base(onnxModelService, promptService, logger)
        {
        }


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
        protected override async Task<DenseTensor<float>> SchedulerStep(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, DenseTensor<float> promptEmbeddings, bool performGuidance, Action<int, int> progressCallback = null, CancellationToken cancellationToken = default)
        {
            using (var scheduler = GetScheduler(schedulerOptions))
            {
                // Get timesteps
                var timesteps = GetTimesteps(schedulerOptions, scheduler);

                // Create latent sample
                var latentsOriginal = await PrepareLatents(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

                // Create masks sample
                var maskImage = PrepareMask(modelOptions, promptOptions, schedulerOptions);

                // Generate some noise
                var noise = scheduler.CreateRandomSample(latentsOriginal.Dimensions);

                // Add noise to original latent
                var latents = scheduler.AddNoise(latentsOriginal, noise, timesteps);

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;
                    var stepTime = Stopwatch.GetTimestamp();
                    cancellationToken.ThrowIfCancellationRequested();

                    // Create input tensor.
                    var inputLatent = performGuidance
                        ? latents.Repeat(2)
                        : latents;
                    var inputTensor = scheduler.ScaleInput(inputLatent, timestep);

                    // Create Input Parameters
                    var inputParameters = CreateUnetInputParams(modelOptions, inputTensor, promptEmbeddings, timestep);

                    // Run Inference
                    using (var inferResult = await _onnxModelService.RunInferenceAsync(modelOptions, OnnxModelType.Unet, inputParameters))
                    {
                        var noisePred = inferResult.FirstElementAs<DenseTensor<float>>();

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

                    progressCallback?.Invoke(step, timesteps.Count);
                    _logger?.LogEnd(LogLevel.Debug, $"Step {step}/{timesteps.Count}", stepTime);
                }

                // Decode Latents
                return await DecodeLatents(modelOptions, promptOptions, schedulerOptions, latents);
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
        protected override async Task<DenseTensor<float>> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            var imageTensor = prompt.InputImage.ToDenseTensor(new[] { 1, 3, options.Height, options.Width });
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.VaeEncoder);
            var outputNames = _onnxModelService.GetOutputNames(model, OnnxModelType.VaeEncoder);

            //TODO: Model Config, Channels
            var outputDim = options.GetScaledDimension();
            var outputBuffer = new DenseTensor<float>(outputDim);
            using (var inputTensorValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, imageTensor.Buffer, imageTensor.Dimensions.ToLong()))
            using (var outputTensorValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, outputBuffer.Buffer, outputDim.ToLong()))
            {
                var inputs = new Dictionary<string, OrtValue> { { inputNames[0], inputTensorValue } };
                var outputs = new Dictionary<string, OrtValue> { { outputNames[0], outputTensorValue } };
                var results = await _onnxModelService.RunInferenceAsync(model, OnnxModelType.VaeEncoder, inputs, outputs);
                using (var result = results.First())
                {
                    var scaledSample = outputBuffer
                       .Add(scheduler.CreateRandomSample(outputBuffer.Dimensions, options.InitialNoiseLevel))
                       .MultiplyBy(model.ScaleFactor);

                    if (prompt.BatchCount > 1)
                        return scaledSample.Repeat(prompt.BatchCount);

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
        private DenseTensor<float> PrepareMask(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            using (var mask = promptOptions.InputImageMask.ToImage())
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
                            var value = pixelSpan[x].A / 255.0f;
                            maskTensor[0, 0, y, x] = 1f - value;
                            maskTensor[0, 1, y, x] = 0f; // Needed for shape only
                            maskTensor[0, 2, y, x] = 0f; // Needed for shape only
                            maskTensor[0, 3, y, x] = 0f; // Needed for shape only
                        }
                    }
                });

                if (promptOptions.BatchCount > 1)
                    return maskTensor.Repeat(promptOptions.BatchCount);

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
            for (int batch = 0; batch < latents.Dimensions[0]; batch++)
            {
                for (int channel = 0; channel < latents.Dimensions[1]; channel++)
                {
                    for (int height = 0; height < latents.Dimensions[2]; height++)
                    {
                        for (int width = 0; width < latents.Dimensions[3]; width++)
                        {
                            float maskValue = mask[batch, 0, height, width];
                            float latentsValue = latents[batch, channel, height, width];
                            float initLatentsProperValue = initLatentsProper[batch, channel, height, width];

                            //Apply the logic to compute the result based on the mask
                            float newValue = initLatentsProperValue * maskValue + latentsValue * (1f - maskValue);
                            result[batch, channel, height, width] = newValue;
                        }
                    }
                }
            }
            return result;
        }
    }
}

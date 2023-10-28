using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Diffusers;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.StableDiffusion.Services
{
    public sealed class InpaintLegacyDiffuser : DiffuserBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InpaintLegacyDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public InpaintLegacyDiffuser(IOnnxModelService onnxModelService, IPromptService promptService)
            : base(onnxModelService, promptService)
        {
        }

        public override async Task<DenseTensor<float>> DiffuseAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();

            // Get Scheduler
            using (var scheduler = GetScheduler(promptOptions, schedulerOptions))
            {
                // Process prompts
                var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, schedulerOptions);

                // Should we perform classifier free guidance
                var performGuidance = schedulerOptions.GuidanceScale > 1.0f;

                // Get timesteps
                var timesteps = GetTimesteps(promptOptions, schedulerOptions, scheduler);

                // Create latent sample
                var latentsOriginal = PrepareLatents(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

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
                        var steplatents = scheduler.Step(noisePred, timestep, latents);

                        // Add noise to original latent
                        var initLatentsProper = scheduler.AddNoise(latentsOriginal, noise, new[] { timestep });

                        // Apply mask and combine 
                        latents = ApplyMaskedLatents(steplatents, initLatentsProper, maskImage);
                    }

                    progress?.Invoke(++step, timesteps.Count);
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
        protected override IReadOnlyList<int> GetTimesteps(PromptOptions prompt, SchedulerOptions options, IScheduler scheduler)
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
        protected override DenseTensor<float> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            // Image input, decode, add noise, return as latent 0
            var imageTensor = prompt.InputImage.ToDenseTensor(new[] { 1, 3, options.Width, options.Height });
            var inputNames = _onnxModelService.GetInputNames(model, OnnxModelType.VaeEncoder);
            var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], imageTensor));
            using (var inferResult = _onnxModelService.RunInference(model, OnnxModelType.VaeEncoder, inputParameters))
            {
                var sample = inferResult.FirstElementAs<DenseTensor<float>>();
                var scaledSample = sample
                     .Add(scheduler.CreateRandomSample(sample.Dimensions, options.InitialNoiseLevel))
                     .MultiplyBy(model.ScaleFactor)
                     .ToDenseTensor();

                if (prompt.BatchCount > 1)
                    return scaledSample.Repeat(prompt.BatchCount);

                return scaledSample;
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
                            var value = (float)pixelSpan[x].A / 255.0f;
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
                            float newValue = (initLatentsProperValue * maskValue) + (latentsValue * (1f - maskValue));
                            result[batch, channel, height, width] = newValue;
                        }
                    }
                }
            }
            return result;
        }
    }
}

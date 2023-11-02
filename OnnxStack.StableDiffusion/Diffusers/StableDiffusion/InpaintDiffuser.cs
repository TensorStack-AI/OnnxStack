using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Helpers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;


namespace OnnxStack.StableDiffusion.Diffusers.StableDiffusion
{
    public sealed class InpaintDiffuser : StableDiffusionDiffuser
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="InpaintDiffuser"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="onnxModelService">The onnx model service.</param>
        public InpaintDiffuser(IOnnxModelService onnxModelService, IPromptService promptService)
            : base(onnxModelService, promptService)
        {
        }


        /// <summary>
        /// Runs the Stable Diffusion inference.
        /// </summary>
        /// <param name="promptOptions">The options.</param>
        /// <param name="schedulerOptions">The scheduler configuration.</param>
        /// <returns></returns>
        public override async Task<DenseTensor<float>> DiffuseAsync(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions, Action<int, int> progress = null, CancellationToken cancellationToken = default)
        {
            // Create random seed if none was set
            schedulerOptions.Seed = schedulerOptions.Seed > 0 ? schedulerOptions.Seed : Random.Shared.Next();

            // Get Scheduler
            using (var scheduler = GetScheduler(promptOptions, schedulerOptions))
            {
                // Should we perform classifier free guidance
                var performGuidance = schedulerOptions.GuidanceScale > 1.0f;

                // Process prompts
                var promptEmbeddings = await _promptService.CreatePromptAsync(modelOptions, promptOptions, performGuidance);

                // Get timesteps
                var timesteps = GetTimesteps(promptOptions, schedulerOptions, scheduler);

                // Create latent sample
                var latents = PrepareLatents(modelOptions, promptOptions, schedulerOptions, scheduler, timesteps);

                // Create Image Mask
                var maskImage = PrepareMask(modelOptions, promptOptions, schedulerOptions);

                // Create Masked Image Latents
                var maskedImage = PrepareImageMask(modelOptions, promptOptions, schedulerOptions);

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
                    inputTensor = ConcatenateLatents(inputTensor, maskedImage, maskImage);

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
                        latents = scheduler.Step(noisePred, timestep, latents).Result;
                    }

                    progress?.Invoke(++step, timesteps.Count);
                }

                // Decode Latents
                return await DecodeLatents(modelOptions, promptOptions, schedulerOptions, latents);
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

                imageTensor = imageTensor.MultiplyBy(modelOptions.ScaleFactor);
                if (promptOptions.BatchCount > 1)
                    imageTensor = imageTensor.Repeat(promptOptions.BatchCount);

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
        /// <param name="scheduler">The scheduler.</param>
        /// <returns></returns>
        private DenseTensor<float> PrepareImageMask(IModelOptions modelOptions, PromptOptions promptOptions, SchedulerOptions schedulerOptions)
        {
            using (var image = promptOptions.InputImage.ToImage())
            using (var mask = promptOptions.InputImageMask.ToImage())
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
                var inputNames = _onnxModelService.GetInputNames(modelOptions, OnnxModelType.VaeEncoder);
                var inputParameters = CreateInputParameters(NamedOnnxValue.CreateFromTensor(inputNames[0], imageMaskedTensor));
                using (var inferResult = _onnxModelService.RunInference(modelOptions, OnnxModelType.VaeEncoder, inputParameters))
                {
                    var sample = inferResult.FirstElementAs<DenseTensor<float>>();
                    var scaledSample = sample.MultiplyBy(modelOptions.ScaleFactor);
                    if (promptOptions.BatchCount > 1)
                        scaledSample = scaledSample.Repeat(promptOptions.BatchCount);

                    if (schedulerOptions.GuidanceScale > 1f)
                        scaledSample = scaledSample.Repeat(2);

                    return scaledSample;
                }
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
        protected override DenseTensor<float> PrepareLatents(IModelOptions model, PromptOptions prompt, SchedulerOptions options, IScheduler scheduler, IReadOnlyList<int> timesteps)
        {
            return scheduler.CreateRandomSample(options.GetScaledDimension(prompt.BatchCount), scheduler.InitNoiseSigma);
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
